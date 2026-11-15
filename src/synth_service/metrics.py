# src/synth_service/metrics.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from synth_service.error_injector import DistortionReport

def ks_metric(real: pd.Series, synth: pd.Series) -> dict[str, float]:
    r = real.dropna().to_numpy()
    s = synth.dropna().to_numpy()
    if len(r) == 0 or len(s) == 0:
        return {"ks": float("nan"), "p_value": float("nan")}
    result = stats.ks_2samp(r, s)
    return {"ks": float(result.statistic), "p_value": float(result.pvalue)}

def tv_metric(real: pd.Series, synth: pd.Series) -> float:
    r = real.dropna().astype(str).value_counts(normalize=True)
    s = synth.dropna().astype(str).value_counts(normalize=True)
    cats = sorted(set(r.index) | set(s.index))
    p = np.array([r.get(c, 0.0) for c in cats])
    q = np.array([s.get(c, 0.0) for c in cats])
    return float(0.5 * np.abs(p - q).sum())

def frobenius_corr_metric(real: pd.DataFrame, synth: pd.DataFrame, cols: list[str]) -> float:
    if not cols:
        return 0.0
    r = real[cols].corr().to_numpy()
    s = synth[cols].corr().to_numpy()
    return float(np.linalg.norm(r - s, "fro"))

def js_metric(real: pd.Series, synth: pd.Series, n_bins: int = 20) -> float:
    r = real.dropna().to_numpy()
    s = synth.dropna().to_numpy()
    if len(r) == 0 or len(s) == 0:
        return float("nan")
    lo = min(r.min(), s.min())
    hi = max(r.max(), s.max())
    if hi == lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(r, bins=bins)
    q, _ = np.histogram(s, bins=bins)
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q) ** 2)

def _to_int64_ns(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce")
    ns = parsed.astype("int64").astype("float64")
    ns = ns.where(parsed.notna(), other=np.nan)
    return ns

def _len_histogram(s: pd.Series, bins: int = 20) -> np.ndarray:
    lengths = s.dropna().astype(str).str.len().to_numpy()
    if lengths.size == 0:
        return np.zeros(bins)
    lo, hi = lengths.min(), lengths.max()
    if hi == lo:
        hist = np.zeros(bins)
        hist[0] = lengths.size
        return hist
    hist, _ = np.histogram(lengths, bins=np.linspace(lo, hi, bins + 1))
    return hist

@dataclass
class IdentifierCheck:
    name: str
    unique_ok: bool
    range_ok: bool
    n_unique: int

@dataclass
class TextCheck:
    name: str
    real_mean_len: float
    synth_mean_len: float
    real_unique: int
    synth_unique: int
    overlap_share: float

@dataclass
class ColumnMetrics:
    name: str
    kind: str
    ks: float | None = None
    ks_p_value: float | None = None
    tv: float | None = None
    js: float | None = None

@dataclass
class ReproductionReport:
    n_rows_real: int
    n_rows_synth: int
    columns: list[ColumnMetrics] = field(default_factory=list)
    frobenius_corr: float | None = None
    numeric_columns_used: list[str] = field(default_factory=list)
    identifier_checks: list[IdentifierCheck] = field(default_factory=list)
    text_checks: list[TextCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _check_identifier(real: pd.Series, synth: pd.Series, name: str) -> IdentifierCheck:
    s = synth.dropna()
    n_unique = int(s.nunique())
    unique_ok = n_unique == len(s)
    range_ok = True
    if pd.api.types.is_numeric_dtype(real) and pd.api.types.is_numeric_dtype(s):
        if len(s) == len(real):
            r_min, r_max = real.min(), real.max()
            s_min, s_max = s.min(), s.max()
            range_ok = bool(s_min >= r_min and s_max <= r_max)
    return IdentifierCheck(name=name, unique_ok=unique_ok, range_ok=range_ok, n_unique=n_unique)

def _check_text(real: pd.Series, synth: pd.Series, name: str) -> TextCheck:
    r = real.dropna().astype(str)
    s = synth.dropna().astype(str)
    real_mean = float(r.str.len().mean()) if len(r) else float("nan")
    synth_mean = float(s.str.len().mean()) if len(s) else float("nan")
    r_set = set(r.tolist())
    s_set = set(s.tolist())
    union = r_set | s_set
    overlap = len(r_set & s_set) / len(union) if union else 0.0
    return TextCheck(
        name=name, real_mean_len=real_mean, synth_mean_len=synth_mean,
        real_unique=len(r_set), synth_unique=len(s_set), overlap_share=float(overlap),
    )

def evaluate_reproduction(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    profile=None,
) -> ReproductionReport:
    common = [c for c in real.columns if c in synth.columns]
    report = ReproductionReport(n_rows_real=len(real), n_rows_synth=len(synth))

    if profile is not None:
        kinds = {c.name: c.kind for c in profile.columns}
    else:
        kinds = {}
        for c in common:
            if pd.api.types.is_numeric_dtype(real[c]):
                kinds[c] = "numeric"
            else:
                kinds[c] = "categorical"

    numeric_used = []
    for name in common:
        kind = kinds.get(name, "categorical")
        cm = ColumnMetrics(name=name, kind=kind)
        if kind == "numeric":
            ks = ks_metric(real[name], synth[name])
            cm.ks = ks["ks"]
            cm.ks_p_value = ks["p_value"]
            cm.js = js_metric(real[name], synth[name])
            numeric_used.append(name)
        elif kind in ("categorical", "boolean"):
            cm.tv = tv_metric(real[name], synth[name])
        elif kind == "identifier":
            report.identifier_checks.append(_check_identifier(real[name], synth[name], name))
        elif kind == "datetime":
            real_ns = _to_int64_ns(real[name])
            synth_ns = _to_int64_ns(synth[name])
            ks = ks_metric(real_ns, synth_ns)
            cm.ks = ks["ks"]
            cm.ks_p_value = ks["p_value"]
            cm.js = js_metric(real_ns, synth_ns)
        elif kind == "text":
            report.text_checks.append(_check_text(real[name], synth[name], name))
            cm.js = js_metric(
                pd.Series(_len_histogram(real[name])),
                pd.Series(_len_histogram(synth[name])),
            )
        report.columns.append(cm)

    report.numeric_columns_used = numeric_used
    report.frobenius_corr = frobenius_corr_metric(real, synth, numeric_used)
    return report

# ============================================================================
# НОВОЕ: Оценка качества искажений (ТЕЗИС: прирост дисперсии, сдвиг корреляционной
# матрицы, совпадение схемы с целевой)
# ============================================================================

# Типы маргинальных искажений, при которых инжектор гарантированно приводит
# целочисленную колонку к float64 (явное приведение в error_injector)
_INT_CASTING_MARGINAL_TYPES = frozenset(
    {"outlier", "numeric_typo", "missing", "noise_additive", "noise_multiplicative"}
)
# Парные действия, переписывающие col2 (float записывается в колонку)
_COL2_REWRITING_PAIRWISE_ACTIONS = frozenset(
    {"break_correlation", "false_link", "invert_correlation", "shift_dependency"}
)

def _dtype_class(s: pd.Series) -> str:
    """Коарс-класс dtype для сравнения схем (int/float/bool/datetime/string/numeric)."""
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "string"

def _target_dtype_class(dtype_str: str | None) -> str:
    """Класс dtype из строки отчета change_type ("float64", "Int64", "datetime64[ns]"...)."""
    if not dtype_str:
        return "string"
    d = str(dtype_str).lower()
    if d.startswith("int"):
        return "integer"
    if d.startswith("float"):
        return "float"
    if d.startswith("datetime"):
        return "datetime"
    if d.startswith("bool"):
        return "bool"
    return "string"

def _expected_schema(
    original_df: pd.DataFrame,
    distortion_report: DistortionReport,
) -> dict[str, str]:
    """Целевая схема: схема original_df с учетом примененной модели искажений.

    Учитываются документированные приведения типов инжектора:
      - маргинальные outlier/numeric_typo/missing/noise_* на int -> float64;
      - missing на bool -> object (bool не хранит NaN);
      - парные действия переписывают col2 -> int-колонка становится float64;
      - distribution_drift на int -> float64;
      - change_type задает целевой dtype, drop_column удаляет колонку,
        rename_column переименовывает.
    Пропущенные (skipped) конфигурации целевую схему не меняют.
    """
    expected = {c: _dtype_class(original_df[c]) for c in original_df.columns}

    for m in distortion_report.applied_marginal_models:
        col, etype = m.get("column"), m.get("type")
        if col not in expected:
            continue
        if expected[col] == "integer" and etype in _INT_CASTING_MARGINAL_TYPES:
            expected[col] = "float"
        elif expected[col] == "bool" and etype == "missing":
            expected[col] = "string"

    for a in distortion_report.applied_pairwise_actions:
        col = a.get("col2")
        if col in expected and expected[col] == "integer" and a.get("action") in _COL2_REWRITING_PAIRWISE_ACTIONS:
            expected[col] = "float"

    for s in distortion_report.structural_changes.get("applied", []):
        act = s.get("action")
        if act == "drop_column":
            expected.pop(s.get("column"), None)
        elif act == "rename_column":
            col, new_name = s.get("column"), s.get("new_name")
            if col in expected and new_name:
                expected[new_name] = expected.pop(col)
        elif act == "change_type":
            col = s.get("column")
            if col in expected:
                expected[col] = _target_dtype_class(s.get("to_dtype"))
        elif act == "distribution_drift":
            col = s.get("column")
            if col in expected and expected[col] == "integer":
                expected[col] = "float"
    return expected

def _schema_match(
    original_df: pd.DataFrame,
    corrupted_df: pd.DataFrame,
    distortion_report: DistortionReport,
) -> tuple[float, bool, list[dict]]:
    """ТЕЗИС: "совпадение схемы с целевой".

    Сравнивает фактическую схему corrupted_df с целевой (expected), построенной
    из исходной схемы и примененной модели искажений. Возвращает
    (score, matches_all, checks): score - доля совпавших колонок,
    matches_all - полное совпадение (включая отсутствие лишних колонок),
    checks - поэлементная детализация.
    """
    expected = _expected_schema(original_df, distortion_report)
    checks: list[dict] = []
    matched = 0
    for col, exp in expected.items():
        if col not in corrupted_df.columns:
            checks.append({
                "column": col, "expected": exp, "actual": "missing",
                "match": False, "note": "колонка отсутствует в результате",
            })
            continue
        actual = _dtype_class(corrupted_df[col])
        ok = actual == exp
        matched += int(ok)
        checks.append({"column": col, "expected": exp, "actual": actual, "match": ok})

    extra_columns = [c for c in corrupted_df.columns if c not in expected]
    for col in extra_columns:
        checks.append({
            "column": col, "expected": None, "actual": _dtype_class(corrupted_df[col]),
            "match": False, "note": "колонки нет в целевой схеме",
        })

    score = matched / len(expected) if expected else 1.0
    matches_all = (matched == len(expected)) and not extra_columns
    return float(score), bool(matches_all), checks

@dataclass
class DistortionEvaluationReport:
    frobenius_corr_shift: float  # Норма Фробениуса разности корреляционных матриц
    variance_growth_avg: float   # Средний относительный прирост дисперсии
    pairwise_targets_met: list[dict] # Выполнение целей по парным искажениям
    # ТЕЗИС: "совпадение схемы с целевой"
    schema_match_score: float = 1.0        # Доля колонок, совпавших с целевой схемой
    schema_matches_target: bool = True     # Полное совпадение схемы с целевой
    schema_checks: list[dict] = field(default_factory=list)  # Поэлементная детализация

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def evaluate_distortion(
    original_df: pd.DataFrame,
    corrupted_df: pd.DataFrame,
    distortion_report: DistortionReport,
) -> DistortionEvaluationReport:
    numeric_cols = [c for c in original_df.columns if pd.api.types.is_numeric_dtype(original_df[c]) and pd.api.types.is_numeric_dtype(corrupted_df[c])]
    
    # 1. Сдвиг корреляционной матрицы (Норма Фробениуса)
    frob_shift = frobenius_corr_metric(original_df, corrupted_df, numeric_cols)
    
    # 2. Прирост дисперсии
    var_growth_list = []
    for col in numeric_cols:
        var_orig = original_df[col].var()
        var_corr = corrupted_df[col].var()
        if var_orig > 1e-9:
            growth = (var_corr - var_orig) / var_orig
            var_growth_list.append(float(growth))
    
    avg_var_growth = float(np.mean(var_growth_list)) if var_growth_list else 0.0

    # 3. Проверка выполнения целей по парным искажениям
    pairwise_met = []
    for action in distortion_report.applied_pairwise_actions:
        if action["action"] in ("break_correlation", "false_link"):
            target = action.get("target_correlation", 0.0)
            new_corr = action.get("new_correlation")
            if new_corr is not None:
                success = abs(new_corr - target) < 0.15  # Допуск 0.15
                pairwise_met.append({
                    "col1": action["col1"],
                    "col2": action["col2"],
                    "action": action["action"],
                    "target": target,
                    "achieved": new_corr,
                    "success": success
                })

    # 4. ТЕЗИС: совпадение схемы с целевой
    schema_score, schema_ok, schema_checks = _schema_match(original_df, corrupted_df, distortion_report)

    return DistortionEvaluationReport(
        frobenius_corr_shift=float(frob_shift),
        variance_growth_avg=avg_var_growth,
        pairwise_targets_met=pairwise_met,
        schema_match_score=float(schema_score),
        schema_matches_target=bool(schema_ok),
        schema_checks=schema_checks,
    )