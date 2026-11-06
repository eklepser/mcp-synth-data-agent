from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_NUMERIC_CANDIDATES = [
    ("norm", stats.norm),
    ("lognorm", stats.lognorm),
    ("gamma", stats.gamma),
    ("expon", stats.expon),
    ("uniform", stats.uniform),
]

_IDENTIFIER_UNIQUE_THRESHOLD = 0.95
_IDENTIFIER_DENSITY_THRESHOLD = 2.0
_HIGH_CARDINALITY_THRESHOLD = 0.5
_INTEGER_SHARE_THRESHOLD = 0.95
_TEXT_UNIQUE_THRESHOLD = 0.5
_TEXT_MIN_MEAN_LEN = 4.0

_DATETIME_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    "%d.%m.%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
]

_DEFAULT_FIRST_NAMES = ["Ivan", "Petr", "Anna", "Maria", "Olga", "Sergey", "Dmitry", "Elena", "Alexey", "Natalia"]
_DEFAULT_LAST_NAMES = ["Ivanov", "Petrov", "Smirnova", "Kuznetsov", "Popova", "Sokolov", "Lebedeva", "Kozlov", "Novikov", "Morozov"]

@dataclass
class ColumnProfile:
    name: str
    kind: str
    n_total: int
    n_missing: int
    missing_fraction: float
    distribution: str | None = None
    dist_params: dict[str, float] = field(default_factory=dict)
    dist_aic: float | None = None
    categories: list[Any] = field(default_factory=list)
    category_freqs: dict[str, float] = field(default_factory=dict)
    high_cardinality: bool = False
    bool_values: list[Any] = field(default_factory=list)
    bool_freqs: dict[str, float] = field(default_factory=dict)
    id_min: float | None = None
    id_max: float | None = None
    id_is_integer: bool = False
    datetime_format: str | None = None
    is_integer: bool = False
    
    text_pattern: str | None = None
    text_options: dict[str, Any] = field(default_factory=dict)
    
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None

@dataclass
class TableProfile:
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    schema: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _detect_datetime_format(s: pd.Series) -> str | None:
    non_null = s.dropna()
    if len(non_null) == 0:
        return None
    sample = non_null.astype(str).head(50)
    for fmt in _DATETIME_FORMATS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, format=fmt, errors="raise")
            if parsed.notna().all():
                return fmt
        except (ValueError, TypeError):
            continue
    return None

def _is_datetime_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not pd.api.types.is_object_dtype(s):
        return False
    non_null = s.dropna()
    if len(non_null) == 0:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(non_null, errors="raise")
        return bool(
            parsed.notna().mean() > 0.99
            and non_null.astype(str).str.contains(r"[-:/]").mean() > 0.5
        )
    except (ValueError, TypeError):
        return False

def _is_dense_integer_identifier(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    clean = s.dropna()
    if len(clean) == 0:
        return False
    vals = clean.astype(float).to_numpy()
    if not np.all(np.equal(np.mod(vals, 1), 0)):
        return False
    n_unique = clean.nunique()
    if n_unique / len(clean) < _IDENTIFIER_UNIQUE_THRESHOLD:
        return False
    span = float(np.max(vals) - np.min(vals))
    if span == 0:
        return False
    return (span / len(clean)) < _IDENTIFIER_DENSITY_THRESHOLD

def _is_integer_column(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    clean = s.dropna()
    if len(clean) == 0:
        return False
    vals = clean.astype(float).to_numpy()
    is_int = np.equal(np.mod(vals, 1), 0)
    return bool(is_int.mean() >= _INTEGER_SHARE_THRESHOLD)

def _looks_like_text(s: pd.Series) -> bool:
    if not pd.api.types.is_object_dtype(s):
        return False
    clean = s.dropna().astype(str)
    if len(clean) == 0:
        return False
    if clean.nunique() / len(clean) < _TEXT_UNIQUE_THRESHOLD:
        return False
    return bool(clean.str.len().mean() >= _TEXT_MIN_MEAN_LEN)

def _classify_series(s: pd.Series) -> str:
    if _is_datetime_like(s):
        return "datetime"

    non_null = s.dropna()
    n_nonnull = len(non_null)
    if n_nonnull == 0:
        return "categorical"

    n_unique = non_null.nunique()
    if 0 < n_unique <= 2:
        return "boolean"

    if _is_dense_integer_identifier(s):
        return "identifier"

    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "numeric"

    if _looks_like_text(s):
        return "text"

    return "categorical"

def _dist_arg_names(dist) -> list[str]:
    shapes = list(getattr(dist, "shapes", "") or "")
    if isinstance(shapes, str):
        shapes = [s.strip() for s in shapes.split(",") if s.strip()]
    return shapes + ["loc", "scale"]

def _fit_numeric(values: np.ndarray) -> tuple[str, dict[str, float], float]:
    best_name, best_params, best_aic = "norm", {}, float("inf")
    for name, dist in _NUMERIC_CANDIDATES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(values)
                loglik = np.sum(dist.logpdf(values, *params))
            if not np.isfinite(loglik):
                continue
            k = len(params)
            aic = 2 * k - 2 * loglik
            if aic < best_aic:
                best_name, best_params, best_aic = name, params, aic
        except Exception:
            continue
    if best_aic == float("inf"):
        mu, sigma = float(np.mean(values)), float(np.std(values) or 1.0)
        best_name, best_params, best_aic = "norm", (mu, sigma), float("inf")
    dist = dict(_NUMERIC_CANDIDATES)[best_name]
    arg_names = _dist_arg_names(dist)
    params_dict = {n: float(p) for n, p in zip(arg_names, best_params)}
    return best_name, params_dict, float(best_aic)

def _extract_text_pattern(strings: list[str]) -> tuple[str, dict[str, Any]]:
    if not strings:
        return "", {}

    # 1. Телефон (начинается с +)
    if all(x.startswith("+") for x in strings):
        return "phone", {"length": len(strings[0])}

    # 2. Email
    if all("@" in x for x in strings):
        domains = set(x.split("@")[-1] for x in strings)
        if len(domains) < 10:
            return "email", {"domain": list(domains)}
        return "email", {"domain": ["yandex.ru", "mail.ru", "gmail.com", "example.com"]}

    # 3. Документ/Код (только цифры, одинаковая длина)
    if all(x.isdigit() for x in strings):
        lengths = set(len(x) for x in strings)
        if len(lengths) == 1:
            return "code", {"length": lengths.pop()}

    # 4. ФИО (2 слова)
    if all(len(x.split()) == 2 for x in strings):
        return "full_name", {"first_name": _DEFAULT_FIRST_NAMES, "last_name": _DEFAULT_LAST_NAMES}

    return "generic", {}

def _parse_str_column(s: pd.Series) -> pd.Series:
    clean = s.dropna()
    if len(clean) == 0:
        return s

    # Пробуем числа, НО с жесткими ограничениями:
    # - не начинается с '0'
    # - не содержит '+' или ' '
    # - не является длинной строкой из цифр (это телефон или документ)
    try:
        sample = clean.iloc[0]
        if isinstance(sample, str):
            if sample.startswith("0") or any(c in sample for c in "+ "):
                raise ValueError("Force string")
            if sample.isdigit() and len(sample) >= 5:
                raise ValueError("Force string (long digits)")
                
            _ = pd.to_numeric(clean, errors="raise")
            return pd.to_numeric(s, errors="raise")
    except (ValueError, TypeError, AttributeError):
        pass

    # Пробуем даты
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(clean, errors="raise")
        if parsed.dt.year.min() > 1900 and parsed.dt.year.max() < 2100:
            return pd.to_datetime(s, errors="coerce")
    except (ValueError, TypeError):
        pass

    return s

def profile_column(s: pd.Series) -> ColumnProfile:
    kind = _classify_series(s)
    n_total = len(s)
    n_missing = int(s.isna().sum())
    cp = ColumnProfile(
        name=str(s.name), kind=kind, n_total=n_total, n_missing=n_missing,
        missing_fraction=n_missing / n_total if n_total else 0.0,
    )

    if kind == "numeric":
        clean = s.dropna().astype(float).to_numpy()
        if clean.size == 0: return cp
        name, params, aic = _fit_numeric(clean)
        cp.distribution, cp.dist_params, cp.dist_aic = name, params, aic
        cp.mean = float(np.mean(clean))
        cp.std = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
        cp.min, cp.max = float(np.min(clean)), float(np.max(clean))
        cp.is_integer = _is_integer_column(s)

    elif kind == "identifier":
        clean = s.dropna()
        vals = clean.astype(float).to_numpy()
        cp.id_min, cp.id_max = float(np.min(vals)), float(np.max(vals))
        cp.id_is_integer = True

    elif kind == "boolean":
        vc = s.dropna().value_counts(normalize=True)
        cp.bool_values = [vc.index[i].item() if hasattr(vc.index[i], "item") else vc.index[i] for i in range(len(vc))]
        cp.bool_freqs = {str(k): float(v) for k, v in vc.items()}

    elif kind == "categorical":
        clean = s.dropna().astype(str)
        vc = clean.value_counts(normalize=True)
        cp.categories = vc.index.tolist()
        cp.category_freqs = {str(k): float(v) for k, v in vc.items()}
        if len(clean) > 0:
            cp.high_cardinality = (clean.nunique() / len(clean)) >= _HIGH_CARDINALITY_THRESHOLD

    elif kind == "text":
        clean = s.dropna().astype(str).tolist()
        cp.text_pattern, cp.text_options = _extract_text_pattern(clean)

    elif kind == "datetime":
        cp.datetime_format = _detect_datetime_format(s)
        clean = pd.to_datetime(s.dropna(), errors="coerce").dropna()
        if len(clean):
            cp.min, cp.max = float(clean.astype("int64").min()), float(clean.astype("int64").max())
    return cp

def profile_table(df: pd.DataFrame) -> TableProfile:
    columns = [profile_column(df[c]) for c in df.columns]
    schema = {c.name: c.kind for c in columns}
    return TableProfile(n_rows=len(df), n_cols=df.shape[1], columns=columns, schema=schema)

def profile_csv(path: str) -> TableProfile:
    df_str = pd.read_csv(path, dtype=str)
    df_parsed = df_str.copy()
    for col in df_str.columns:
        df_parsed[col] = _parse_str_column(df_str[col])
    return profile_table(df_parsed)