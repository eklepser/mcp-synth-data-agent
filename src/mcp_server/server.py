import json
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from mcp.server.fastmcp import FastMCP

from synth_service.profile import profile_csv, _parse_str_column
from synth_service.generate import generate_table
from synth_service.error_injector import error_model_from_dict, inject_errors
from synth_service.metrics import evaluate_reproduction, evaluate_distortion

mcp = FastMCP("SynthDataService")


def _clean_nan(obj):
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _safe_json_dumps(data, **kwargs) -> str:
    return json.dumps(_clean_nan(data), **kwargs, allow_nan=False)


@mcp.tool()
def list_available_files() -> str:
    """Возвращает список доступных CSV-файлов в папке data/raw/ и конфигов в data/configs/."""
    try:
        raw_dir = Path("data/raw")
        cfg_dir = Path("data/configs")
        
        raw_files = [f.name for f in raw_dir.glob("*.csv")] if raw_dir.exists() else []
        cfg_files = [f.name for f in cfg_dir.glob("*.json")] if cfg_dir.exists() else []
        
        return _safe_json_dumps({
            "source_files": raw_files,
            "config_files": cfg_files
        }, indent=2)
    except Exception as e:
        return _safe_json_dumps({"error": str(e)})


@mcp.tool()
def profile_data(csv_path: str) -> str:
    """Профилирует CSV и возвращает JSON-строку с типами колонок."""
    try:
        prof = profile_csv(csv_path)
        return _safe_json_dumps(prof.to_dict(), indent=2, ensure_ascii=False)
    except Exception as e:
        return _safe_json_dumps({"error": str(e)})


def _get_combined_stats(df: pd.DataFrame, synth: pd.DataFrame, eval_report):
    rows = []
    if eval_report:
        for c in eval_report.columns:
            r_clean = df[c.name].dropna()
            s_clean = synth[c.name].dropna()
            
            def get_row_stats(s, kind, prefix):
                if kind in ("numeric", "integer"):
                    return {
                        "column": f"{prefix}{c.name}",
                        "kind": kind,
                        "n": len(s),
                        "mean": float(s.mean()) if len(s) else None,
                        "std": float(s.std(ddof=1)) if len(s) > 1 else None,
                        "min": float(s.min()) if len(s) else None,
                        "max": float(s.max()) if len(s) else None,
                        "top": None,
                        "top_share": None,
                        "uniq": int(s.nunique()),
                        "is_unique": int(s.nunique()) == len(s)
                    }
                else:
                    val_counts = s.astype(str).value_counts(normalize=True)
                    top_val = str(val_counts.index[0]) if len(val_counts) > 0 else None
                    top_share = float(val_counts.iloc[0]) if len(val_counts) > 0 else None
                    return {
                        "column": f"{prefix}{c.name}",
                        "kind": kind,
                        "n": len(s),
                        "mean": None,
                        "std": None,
                        "min": None,
                        "max": None,
                        "top": top_val,
                        "top_share": top_share,
                        "uniq": int(s.nunique()),
                        "is_unique": int(s.nunique()) == len(s)
                    }

            rows.append(get_row_stats(r_clean, c.kind, ""))
            rows.append(get_row_stats(s_clean, c.kind, ""))
    return rows


@mcp.tool()
def generate_dataset(
    source_path: str,
    rows: int = 100,
    target_similarity: float = 0.9,
    max_attempts: int = 5,
    random_state: int = 42
) -> str:
    """Генерирует чистую синтетику, стремясь к заданному сходству (similarity)."""
    try:
        if not str(source_path).startswith("data/"):
            source_path = f"data/raw/{source_path}"

        df_str = pd.read_csv(source_path, dtype=str)
        df = df_str.copy()
        for col in df_str.columns:
            df[col] = _parse_str_column(df_str[col])

        prof = profile_csv(source_path)
        
        best_synth = None
        best_report = None
        best_score = -1.0

        for attempt in range(max_attempts):
            rs = random_state + attempt
            synth = generate_table(df, prof, n=rows, random_state=rs)
            report = evaluate_reproduction(df, synth, profile=prof)
            
            ks_vals = [c.ks for c in report.columns if c.ks is not None]
            tv_vals = [c.tv for c in report.columns if c.tv is not None]
            parts = []
            if ks_vals: parts.append(1.0 - sum(ks_vals)/len(ks_vals))
            if tv_vals: parts.append(1.0 - sum(tv_vals)/len(tv_vals))
            if report.frobenius_corr is not None: parts.append(1.0 / (1.0 + report.frobenius_corr))
            score = sum(parts)/len(parts) if parts else 0.0
            
            if score > best_score:
                best_score = score
                best_synth = synth
                best_report = report
                
            if score >= target_similarity:
                break

        out_path = f"data/generated/{Path(source_path).stem}_synth.csv"
        best_synth.to_csv(out_path, index=False)

        sample_df = best_synth.head(50)
        if "id" in sample_df.columns:
            try:
                sample_df = sample_df.sort_values("id")
            except Exception:
                pass
        sample_data = _clean_nan(sample_df.head(10).to_dict(orient="records"))

        result = {
            "source_file": str(source_path),
            "generated_csv": str(out_path),
            "generated_rows": len(best_synth),
            "target_similarity": target_similarity,
            "achieved_similarity": best_score,
            "mode": "clean",
            "column_stats": _get_combined_stats(df, best_synth, best_report),
            "sample_data": sample_data
        }
        return _safe_json_dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return _safe_json_dumps({"error": f"Не удалось сгенерировать данные: {str(e)}"})


@mcp.tool()
def generate_custom_distortions(
    source_path: str,
    rows: int,
    rules_json: str,
    random_state: int = 42
) -> str:
    """Генерирует синтетику и применяет искажения заданного типа, интенсивности и адресации.

    rules_json - JSON со спецификацией искажений. Допустимы два формата.

    1) ПОЛНЫЙ (рекомендуется): {"marginal": {...}, "pairwise": [...], "structural": [...]}

       "marginal" - объект {колонка: {параметры}}; параметры:
         error_type (обязателен): typo | numeric_typo | outlier | missing | category_swap |
                                  noise_additive | noise_multiplicative | precision_loss
         error_rate: доля искажаемых ячеек, число в [0, 1]
         mechanism: "ecar" (по умолчанию) - ячейки выбираются случайно;
                    "enar" - только строки, удовлетворяющие condition
         condition: pandas-выражение по ЛЮБЫМ колонкам таблицы для mechanism="enar",
                    например "age > 40 && city == 'Moscow'"
         intensity: сила искажения - для noise_additive/noise_multiplicative сигма шума
                    как доля std колонки; для precision_loss - сколько знаков после
                    запятой оставить (по умолчанию 1)

       "pairwise" - список объектов для парных искажений (корреляция двух колонок):
         {"col1", "col2", "action": "break_correlation" | "invert_correlation" | "shift_dependency"}
         Для "break_correlation" можно задать "target_correlation" в [-1, 1]:
           - 0.0 (по умолчанию) - полный разрыв через перестановку col2;
           - любое значение в (-1, 1) - ослабление/усиление до целевой корреляции,
             сохраняя маргинальное mean/std col2.
         Для "shift_dependency" обязательно "target_slope" - новый наклон линейной
         зависимости col2 = a*col1 + b + eps (сохраняет mean/std col2, меняет форму связи).

       "structural" - список объектов для искажений таблицы в целом:
         {"action": "duplicate_rows", "rate": 0.1} - продублировать rate-долю строк
         {"action": "drop_column", "columns": ["name"]} - удалить колонки
         {"action": "rename_column", "columns": ["old"], "new_name": "new"}
         {"action": "change_type", "columns": ["age"], "new_type": "float|int|string|datetime"}
         {"action": "distribution_drift", "columns": ["age"], "delta": 5.0, "scale": 1.2}
           - дрейф распределения: Y' = (Y + delta) * scale
           (delta - аддитивный сдвиг, scale - мультипликативный масштаб, scale > 0).

    2) ПЛОСКИЙ (совместимость): {колонка: {"error_type", "rate", "mechanism", "condition",
       "intensity"}} - то же, что "marginal" без обёртки, ключ "rate" равнозначен "error_rate".

    В ответе: distortion_report (план/факт по каждой модели, предупреждения, пропуски,
    структурные изменения) и applied_model - итоговая применённая модель искажений.
    """
    try:
        if not str(source_path).startswith("data/"):
            source_path = f"data/raw/{source_path}"

        rules = json.loads(rules_json)
        if not isinstance(rules, dict):
            return _safe_json_dumps({"error": "rules_json должен быть JSON-объектом"})

        # Совместимость со старым форматом тула: плоский {колонка: {..., "rate": ...}}
        # нормализуем в {"marginal": ...} (ключ rate -> error_rate) и далее единый путь.
        if "marginal" not in rules and "pairwise" not in rules and "structural" not in rules:
            flat = {}
            for col, rule in rules.items():
                if not isinstance(rule, dict):
                    return _safe_json_dumps(
                        {"error": f"Правило для колонки '{col}' должно быть JSON-объектом"}
                    )
                flat[col] = {("error_rate" if k == "rate" else k): v for k, v in rule.items()}
            rules = {"marginal": flat}

        try:
            model = error_model_from_dict(rules)
        except (ValueError, TypeError) as e:
            return _safe_json_dumps({"error": f"Некорректная модель искажений: {e}"})

        df_str = pd.read_csv(source_path, dtype=str)
        df = df_str.copy()
        for col in df_str.columns:
            df[col] = _parse_str_column(df_str[col])

        prof = profile_csv(source_path)
        synth_clean = generate_table(df, prof, n=rows, random_state=random_state)

        synth_noisy, mask, report = inject_errors(synth_clean, model, random_state=random_state)

        out_path = f"data/generated/{Path(source_path).stem}_noisy.csv"
        synth_noisy.to_csv(out_path, index=False)

        eval_report = evaluate_reproduction(df, synth_noisy, profile=prof)

        # ТЕЗИС: "В режиме искажения измеряется соответствие фактического отклонения
        # заданному: прирост дисперсии, сдвиг корреляционной матрицы, совпадение схемы
        # с целевой." Без этого вызова у LLM-агента нет метрик режима искажения.
        distortion_eval = evaluate_distortion(synth_clean, synth_noisy, report)

        sample_df = synth_noisy.head(50)
        if "id" in sample_df.columns:
            try:
                sample_df = sample_df.sort_values("id")
            except Exception:
                pass
        sample_data_clean = _clean_nan(sample_df.head(10).to_dict(orient="records"))

        result = {
            "source_file": str(source_path),
            "generated_csv": str(out_path),
            "generated_rows": len(synth_noisy),
            "mode": "noisy",
            "distortion_report": report.to_dict(),
            "distortion_evaluation": distortion_eval.to_dict(),
            "applied_model": model.to_dict(),
            "column_stats": _get_combined_stats(df, synth_noisy, eval_report),
            "sample_data": sample_data_clean
        }
        return _safe_json_dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return _safe_json_dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()