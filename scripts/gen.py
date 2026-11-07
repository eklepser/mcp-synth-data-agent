"""Unified generation entry point."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from synth_service.profile import profile_csv, _parse_str_column
from synth_service.generate import generate_table
from synth_service.metrics import evaluate_reproduction

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic dataset from a CSV file.")
    p.add_argument("source", help="Path to the source CSV file")
    p.add_argument("--rows", type=int, default=None, help="Number of rows to generate (default: same as source)")
    p.add_argument("--similarity", type=float, default=0.9, help="Target similarity level in [0, 1] (default: 0.9)")
    p.add_argument("--random-state", type=int, default=42, help="Base random seed (default: 42)")
    p.add_argument("--max-attempts", type=int, default=5, help="Max regeneration attempts (default: 5)")
    p.add_argument("--out-dir", default="data/generated", help="Output directory (default: data/generated)")
    p.add_argument("--clip-to-range", action="store_true", help="Clip generated values to source min/max per numeric column")
    p.add_argument("--nv", "--no-visualize", dest="no_visualize", action="store_true", help="Do not open the visualization window")
    return p

def similarity_score(report, source_df: pd.DataFrame, synth_df: pd.DataFrame, profile) -> float:
    ks_vals = [c.ks for c in report.columns if c.kind in ("numeric", "datetime") and c.ks is not None]
    tv_vals = [c.tv for c in report.columns if c.kind in ("categorical", "boolean") and c.tv is not None]
    
    # Сравниваем mean и std для числовых столбцов
    stats_errors = []
    for c in profile.columns:
        if c.kind == "numeric" and c.name in source_df.columns and c.name in synth_df.columns:
            real = source_df[c.name].dropna()
            syn = synth_df[c.name].dropna()
            if len(real) > 1 and len(syn) > 1:
                std_err = abs(real.std() - syn.std()) / (real.std() + 1e-9)
                mean_err = abs(real.mean() - syn.mean()) / (abs(real.mean()) + 1e-9)
                stats_errors.extend([std_err, mean_err])

    parts = []
    if ks_vals: parts.append(1.0 - float(np.mean(ks_vals)))
    if tv_vals: parts.append(1.0 - float(np.mean(tv_vals)))
    if stats_errors: parts.append(1.0 - float(np.mean(stats_errors))) # Ошибка статистик
    if report.frobenius_corr is not None: parts.append(1.0 / (1.0 + float(report.frobenius_corr)))
    
    return float(np.mean(parts)) if parts else 0.0

def describe_columns(df: pd.DataFrame, profile) -> list[dict]:
    kinds = {c.name: c.kind for c in profile.columns}
    is_ints = {c.name: c.is_integer for c in profile.columns}
    rows = []
    for col in df.columns:
        kind = kinds.get(col, "categorical")
        if kind == "numeric" and is_ints.get(col, False):
            kind = "integer"
            
        s = df[col]
        n_notna = int(s.notna().sum())
        n_unique = int(s.dropna().nunique())
        
        entry = {
            "name": col, "kind": kind, "n": n_notna,
            "mean": None, "std": None, "min": None, "max": None, "top": None, "top_share": None,
            "uniq": n_unique, "is_unique": n_unique == n_notna
        }
        if kind in ("numeric", "integer"):
            clean = s.dropna()
            if len(clean):
                entry.update(mean=float(clean.mean()), std=float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
                             min=float(clean.min()), max=float(clean.max()))
        elif kind in ("categorical", "boolean"):
            clean = s.dropna().astype(str)
            if len(clean):
                vc = clean.value_counts(normalize=True)
                entry.update(top=str(vc.index[0]), top_share=float(vc.iloc[0]))
        elif kind == "datetime":
            clean = pd.to_datetime(s.dropna(), errors="coerce").dropna()
            if len(clean):
                entry.update(min=clean.min().strftime("%Y-%m-%d"), max=clean.max().strftime("%Y-%m-%d"))
        elif kind in ("text", "identifier"):
            clean = s.dropna()
            if pd.api.types.is_numeric_dtype(clean) and len(clean):
                entry.update(min=float(clean.min()), max=float(clean.max()))
            else:
                entry.update(top=str(clean.iloc[0])[:20] if len(clean) else None)
        rows.append(entry)
    return rows

def build_metrics_dict(source: Path, out_csv: Path, rows: int, target: float, achieved: float,
                       attempts: int, report, source_df: pd.DataFrame, synth_df: pd.DataFrame, profile) -> dict:
    return {
        "source_file": str(source), "source_csv": str(source), "generated_csv": str(out_csv),
        "generated_rows": rows, "target_similarity": target, "achieved_similarity": achieved,
        "attempts_used": attempts, "source_stats": describe_columns(source_df, profile),
        "generated_stats": describe_columns(synth_df, profile),
        "columns": [{"name": c.name, "kind": c.kind, "ks": c.ks, "ks_p_value": c.ks_p_value, "tv": c.tv, "js": c.js} for c in report.columns],
        "frobenius_corr": report.frobenius_corr,
        "identifier_checks": [asdict(chk) for chk in report.identifier_checks],
        "text_checks": [asdict(chk) for chk in report.text_checks],
        "warnings": []
    }

def main() -> int:
    args = build_parser().parse_args()
    source = Path(args.source)
    if not source.exists():
        print(f"Error: source file not found: {source}", file=sys.stderr)
        return 2

    df_str = pd.read_csv(source, dtype=str)
    df = df_str.copy()
    for col in df_str.columns:
        df[col] = _parse_str_column(df_str[col])

    prof = profile_csv(str(source))
    rows = args.rows if args.rows is not None else len(df)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    out_csv = out_dir / f"{stem}_generated.csv"
    out_json = out_dir / f"{stem}_metrics.json"

    best_synth, best_report, best_score, attempts_used = None, None, -1.0, 0

    for attempt in range(1, args.max_attempts + 1):
        attempts_used = attempt
        rs = args.random_state + (attempt - 1)
        synth = generate_table(df, prof, n=rows, random_state=rs)

        if args.clip_to_range:
            for c in prof.columns:
                if c.kind == "numeric" and c.min is not None and c.max is not None and c.name in synth.columns:
                    synth[c.name] = synth[c.name].clip(lower=c.min, upper=c.max)

        report = evaluate_reproduction(df, synth, profile=prof)
        score = similarity_score(report, df, synth, prof)

        if score > best_score:
            best_score, best_synth, best_report = score, synth, report

        if score >= args.similarity:
            break

    best_synth.to_csv(out_csv, index=False)
    metrics = build_metrics_dict(source, out_csv, rows, args.similarity, best_score, attempts_used, best_report, df, best_synth, prof)
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved CSV:     {out_csv}")
    print(f"Saved metrics: {out_json}")

    if not args.no_visualize:
        vis_script = Path(__file__).parent / "visualize.py"
        subprocess.Popen([sys.executable, str(vis_script), str(out_json)])

    return 0

if __name__ == "__main__":
    sys.exit(main())