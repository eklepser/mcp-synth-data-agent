from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

from synth_service.profile import TableProfile, _classify_series

# ============================================================================
# 1. ТИПЫ ОШИБОК (ErrorType)
# ============================================================================

def apply_typo(value: str, rng: random.Random) -> str:
    if not isinstance(value, str) or len(value) < 2:
        return value
    idx = rng.randint(0, len(value) - 2)
    chars = list(value)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)

def _shift(value: float, rng: random.Random) -> float:
    return value * rng.choice([0.1, 10.0])

def apply_numeric_typo(value: float, rng: random.Random) -> float:
    s = str(value)
    kind = rng.choice(["shift", "drop_zero", "duplicate_digit"])
    if kind == "shift":
        return _shift(value, rng)
    if kind == "drop_zero":
        if "0" in s:
            new_value = float(s.replace("0", "", 1))
            if new_value != value:
                return new_value
        return _shift(value, rng)
    digit_positions = [i for i, ch in enumerate(s) if ch.isdigit()]
    if not digit_positions:
        return _shift(value, rng)
    idx = rng.choice(digit_positions)
    new_value = float(s[:idx] + s[idx] + s[idx:])
    if new_value != value:
        return new_value
    return _shift(value, rng)

def apply_outlier(value: float, rng: random.Random) -> float:
    multiplier = rng.choice([0.01, 0.1, 10.0, 100.0, -1.0])
    return float(value * multiplier)

def apply_missing(_: Any, __: random.Random) -> float:
    return np.nan

def apply_category_swap(value: Any, rng: random.Random, valid_categories: list) -> Any:
    alternatives = [v for v in valid_categories if v != value]
    if not alternatives:
        return value
    return rng.choice(alternatives)

ERROR_TYPE_FUNCS = {
    "typo": apply_typo,
    "numeric_typo": apply_numeric_typo,
    "outlier": apply_outlier,
    "missing": apply_missing,
    "category_swap": apply_category_swap,
}

VECTORIZED_ERROR_TYPES = ("noise_additive", "noise_multiplicative", "precision_loss")

MARGINAL_ERROR_TYPES = set(ERROR_TYPE_FUNCS) | set(VECTORIZED_ERROR_TYPES)

NUMERIC_ONLY_ERROR_TYPES = {
    "numeric_typo", "outlier",
    "noise_additive", "noise_multiplicative", "precision_loss",
}

# ============================================================================
# 2. МЕХАНИЗМЫ (ErrorMechanism)
# ============================================================================

def get_ecar_mask(n: int, error_rate: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n) < error_rate

def _eval_enar_condition(df: pd.DataFrame, condition: str) -> np.ndarray:
    try:
        evaluated = df.eval(condition)
    except Exception as e:
        raise ValueError(
            f"Не удалось оценить ENAR-условие '{condition}' "
            f"(колонки фрейма: {list(df.columns)}): {e}"
        ) from e
    if isinstance(evaluated, pd.Series):
        return evaluated.fillna(False).astype(bool).to_numpy()
    return np.asarray(evaluated, dtype=bool)

def get_enar_mask(df: pd.DataFrame, error_rate: float, condition: str,
                  rng: np.random.Generator) -> np.ndarray:
    """ENAR: вероятность ошибки зависит от значений столбцов таблицы [1]."""
    base_mask = rng.random(len(df)) < error_rate
    return base_mask & _eval_enar_condition(df, condition)

# ============================================================================
# 3. МОДЕЛЬ ОШИБОК И ОТЧЕТ (ErrorModel & DistortionReport)
# ============================================================================

@dataclass
class ColumnErrorConfig:
    error_type: Literal["typo", "numeric_typo", "outlier", "missing", "category_swap",
                        "noise_additive", "noise_multiplicative", "precision_loss"]
    error_rate: float
    mechanism: Literal["ecar", "enar"] = "ecar"
    condition: str | None = None
    intensity: float | None = None
    matched_by: str | None = None  # внутреннее поле: селектор/группа, из которой получено правило

    def __post_init__(self):
        if isinstance(self.error_rate, bool) or not isinstance(self.error_rate, (int, float)) \
                or not (0.0 <= float(self.error_rate) <= 1.0):
            raise ValueError(f"error_rate должен быть числом в [0, 1], получено: {self.error_rate!r}")
        if self.error_type not in MARGINAL_ERROR_TYPES:
            raise ValueError(
                f"Неизвестный тип ошибки: {self.error_type!r}. Доступны: {sorted(MARGINAL_ERROR_TYPES)}"
            )
        if self.mechanism not in ("ecar", "enar"):
            raise ValueError(f"mechanism должен быть 'ecar' или 'enar', получено: {self.mechanism!r}")
        if self.mechanism == "enar" and not self.condition:
            raise ValueError("mechanism='enar' требует condition - pandas-выражение по колонкам таблицы")

@dataclass
class PairwiseDistortionConfig:
    col1: str
    col2: str
    action: Literal["break_correlation", "false_link", "invert_correlation", "shift_dependency"]

    method: Literal["rearrangement", "mixture"] = "rearrangement"
    target_correlation: float | None = None
    target_slope: float | None = None  

    def __post_init__(self):
        if self.action not in ("break_correlation", "false_link", "invert_correlation", "shift_dependency"):
            raise ValueError(
                f"Неизвестное парное действие: {self.action!r}. "
                f"Доступны: break_correlation, false_link, invert_correlation, shift_dependency"
            )
        if self.action in ("break_correlation", "false_link"):
            if self.target_correlation is None:
                self.target_correlation = 0.5 if self.action == "false_link" else 0.0
            if not (-1.0 <= float(self.target_correlation) <= 1.0):
                raise ValueError(
                    f"target_correlation должен быть в [-1, 1], получено: {self.target_correlation!r}"
                )
        if self.action == "shift_dependency" and self.target_slope is None:
            raise ValueError("shift_dependency требует target_slope (числовой коэффициент наклона)")
        if self.method not in ("rearrangement", "mixture"):
            raise ValueError(
                f"method должен быть 'rearrangement' или 'mixture', получено: {self.method!r}"
            )

STRUCTURAL_ACTIONS = ("duplicate_rows", "drop_column", "rename_column", "change_type",
                      "distribution_drift", "violate_constraints")
CHANGE_TYPE_TARGETS = ("float", "int", "string", "datetime")

CONSTRAINT_TARGETS = ("unique", "range", "not_null")

@dataclass
class StructuralDistortionConfig:
    action: Literal["duplicate_rows", "drop_column", "rename_column", "change_type",
                    "distribution_drift", "violate_constraints"]
    columns: list[str] = field(default_factory=list)
    new_name: str | None = None
    new_type: str | None = None
    rate: float = 0.05
    delta: float = 0.0   
    scale: float = 1.0   
    constraint: str | None = None  
    min_value: float | None = None 
    max_value: float | None = None 

    def __post_init__(self):
        if self.action not in STRUCTURAL_ACTIONS:
            raise ValueError(
                f"Неизвестное структурное действие: {self.action!r}. Доступны: {list(STRUCTURAL_ACTIONS)}"
            )
        if self.action == "duplicate_rows" and not (0.0 <= float(self.rate) <= 1.0):
            raise ValueError(f"rate для duplicate_rows должен быть в [0, 1], получено: {self.rate!r}")
        if self.action == "drop_column" and not self.columns:
            raise ValueError("drop_column требует непустой список columns")
        if self.action == "rename_column" and (len(self.columns) != 1 or not self.new_name):
            raise ValueError("rename_column требует ровно одну колонку в columns и new_name")
        if self.action == "change_type" and (not self.columns or self.new_type not in CHANGE_TYPE_TARGETS):
            raise ValueError(f"change_type требует columns=[...] и new_type из {list(CHANGE_TYPE_TARGETS)}")
        if self.action == "distribution_drift":
            if len(self.columns) != 1:
                raise ValueError("distribution_drift требует ровно одну колонку в columns")
            if not (isinstance(self.delta, (int, float)) and not isinstance(self.delta, bool)):
                raise ValueError(f"delta должен быть числом, получено: {self.delta!r}")
            if not (isinstance(self.scale, (int, float)) and not isinstance(self.scale, bool)) or float(self.scale) <= 0:
                raise ValueError(f"scale должен быть положительным числом, получено: {self.scale!r}")
        if self.action == "violate_constraints":
            if self.constraint not in CONSTRAINT_TARGETS:
                raise ValueError(
                    f"constraint должен быть из {list(CONSTRAINT_TARGETS)}, получено: {self.constraint!r}"
                )
            if not self.columns:
                raise ValueError("violate_constraints требует непустой список columns")
            if self.constraint == "range":
                if len(self.columns) != 1:
                    raise ValueError("constraint='range' требует ровно одну колонку в columns")
                for bound_name, bound in (("min_value", self.min_value), ("max_value", self.max_value)):
                    if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float))):
                        raise ValueError(f"{bound_name} должен быть числом, получено: {bound!r}")
                if self.min_value is None and self.max_value is None:
                    raise ValueError(
                        "constraint='range' требует min_value и/или max_value "
                        "(заявленный допустимый диапазон колонки)"
                    )

@dataclass
class DistortionReport:
    target_error_rate: float
    actual_error_rate: float
    corrupted_cells_count: int
    applied_marginal_models: list[dict] = field(default_factory=list)
    applied_pairwise_actions: list[dict] = field(default_factory=list)
    structural_changes: dict[str, Any] = field(default_factory=dict)
    planned_cells_count: int = 0
    no_op_cells_count: int = 0
    warnings: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ErrorModel:
    column_configs: dict[str, ColumnErrorConfig] = field(default_factory=dict)
    pairwise_configs: list[PairwiseDistortionConfig] = field(default_factory=list)
    structural_configs: list[StructuralDistortionConfig] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {"marginal": {}, "pairwise": [], "structural": []}
        for col, cfg in self.column_configs.items():
            result["marginal"][col] = asdict(cfg)
        for cfg in self.pairwise_configs:
            result["pairwise"].append(asdict(cfg))
        for cfg in self.structural_configs:
            result["structural"].append(asdict(cfg))
        return result

def _build_column_config(col: str, cfg_dict: dict) -> ColumnErrorConfig:
    if not isinstance(cfg_dict, dict):
        raise ValueError(
            f"Конфигурация колонки '{col}' должна быть JSON-объектом, получено: {type(cfg_dict).__name__}"
        )
    cfg = dict(cfg_dict)
    cfg.pop("matched_by", None)  # внутреннее поле provenance, в пользовательских правилах игнорируется
    if "rate" in cfg and "error_rate" not in cfg:
        cfg["error_rate"] = cfg.pop("rate")
    unknown = set(cfg) - set(ColumnErrorConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Конфигурация колонки '{col}' содержит неизвестные поля: {sorted(unknown)}")
    try:
        return ColumnErrorConfig(**cfg)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Конфигурация колонки '{col}': {e}") from e

# ============================================================================
# 3.1 АДРЕСАЦИЯ МАРГИНАЛЬНЫХ ПРАВИЛ: столбец / группа столбцов / таблица
# ============================================================================

COLUMN_KINDS = ("numeric", "categorical", "boolean", "text", "datetime", "identifier")


def _classify_columns(df: pd.DataFrame) -> dict[str, str]:
    """Тип каждой колонки по классификации профайлера (без подбора распределений)."""
    return {c: _classify_series(df[c]) for c in df.columns}


def _resolve_selector(key: str, df: pd.DataFrame | None,
                      kinds: dict[str, str]) -> list[str]:
    """Разворачивает селектор '*' (все колонки) или '*<тип>' в список колонок."""
    if df is None:
        raise ValueError(
            f"Селектор '{key}' требует передачи данных (df): соответствие колонок "
            f"определяется по фактической таблице"
        )
    if key == "*":
        return list(df.columns)
    kind = key[1:]
    if kind not in COLUMN_KINDS:
        raise ValueError(
            f"Неизвестный селектор '{key}'. Доступны: '*' и "
            f"{', '.join('*' + k for k in COLUMN_KINDS)}"
        )
    cols = [c for c, k in kinds.items() if k == kind]
    if not cols:
        raise ValueError(
            f"Селектор '{key}' не соответствует ни одной колонке таблицы; "
            f"фактические типы колонок: {kinds}"
        )
    return cols


def _iter_marginal_rules(
    marginal_data: Any,
    df: pd.DataFrame | None,
    kinds: dict[str, str],
) -> Iterator[tuple[str, dict, str | None]]:
    """Итерирует правила секции 'marginal' в порядке следования.

    Формы адресации («столбец, группа столбцов, таблица»):
      1. Явная: {"<колонка>": {...}};
      2. Селекторы: "*" - все колонки таблицы; "*numeric"/"*categorical"/
         "*boolean"/"*text"/"*datetime"/"*identifier" - все колонки данного
         типа (тип определяется профайлером по фактическому df);
      3. Список правил: [{"columns": [...], ...}, ...] (или "column": "...").
    Возвращает тройки (колонка, raw-config, matched_by). При пересечении
    правил действует последнее в порядке следования - так задаются
    исключения (сначала группа, затем правило для конкретной колонки).
    """
    if isinstance(marginal_data, dict):
        for key, cfg_dict in marginal_data.items():
            if not isinstance(cfg_dict, dict):
                raise ValueError(
                    f"Правило '{key}' должно быть JSON-объектом, "
                    f"получено: {type(cfg_dict).__name__}"
                )
            if key.startswith("*"):
                for col in _resolve_selector(key, df, kinds):
                    yield col, cfg_dict, key
            else:
                yield key, cfg_dict, None
    elif isinstance(marginal_data, list):
        for i, entry in enumerate(marginal_data):
            if not isinstance(entry, dict):
                raise ValueError(f"Элемент marginal[{i}] должен быть JSON-объектом")
            entry = dict(entry)
            cols = entry.pop("columns", None)
            if cols is None:
                single = entry.pop("column", None)
                cols = [single] if single is not None else None
            if not cols or not isinstance(cols, list) \
                    or not all(isinstance(c, str) for c in cols):
                raise ValueError(
                    f"Элемент marginal[{i}] должен содержать непустой список строк "
                    f"'columns' (или одиночную строку 'column')"
                )
            matched_by = f"columns[{','.join(cols)}]"
            for col in cols:
                yield col, entry, matched_by
    else:
        raise ValueError(
            "Секция 'marginal' должна быть JSON-объектом {колонка/селектор: правило} "
            "или списком правил [{'columns': [...], ...}]"
        )


def _build_marginal_configs(marginal_data: Any,
                            df: pd.DataFrame | None) -> dict[str, ColumnErrorConfig]:
    """Разворачивает правила в плоский словарь {колонка: ColumnErrorConfig}."""
    kinds = _classify_columns(df) if df is not None else {}
    resolved: dict[str, tuple[dict, str | None]] = {}
    for col, cfg_dict, matched_by in _iter_marginal_rules(marginal_data, df, kinds):
        resolved[col] = (cfg_dict, matched_by)
    configs: dict[str, ColumnErrorConfig] = {}
    for col, (cfg_dict, matched_by) in resolved.items():
        config = _build_column_config(col, cfg_dict)
        config.matched_by = matched_by
        configs[col] = config
    return configs


def error_model_from_dict(data: dict, df: pd.DataFrame | None = None) -> ErrorModel:
    """Строит ErrorModel из словаря конфигурации.

    df (опционально) нужен для групповой адресации marginal: селекторы '*',
    '*numeric', '*categorical', '*boolean', '*text', '*datetime', '*identifier'
    разворачиваются по фактическим типам колонок таблицы. Для явных правил
    {колонка: {...}} df не требуется.
    """
    if not isinstance(data, dict):
        raise ValueError("Конфигурация ошибок должна быть JSON-объектом")

    if "marginal" in data:
        marginal_data = data.get("marginal") or {}
    elif "pairwise" in data or "structural" in data:
        marginal_data = {}
    else:
        marginal_data = data

    if marginal_data:
        configs = _build_marginal_configs(marginal_data, df)
    else:
        configs = {}

    pairwise = []
    for i, p in enumerate(data.get("pairwise") or []):
        try:
            pairwise.append(PairwiseDistortionConfig(**p))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Конфигурация pairwise[{i}]: {e}") from e

    structural = []
    for i, s in enumerate(data.get("structural") or []):
        try:
            structural.append(StructuralDistortionConfig(**s))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Конфигурация structural[{i}]: {e}") from e

    return ErrorModel(column_configs=configs, pairwise_configs=pairwise, structural_configs=structural)

def load_error_model(json_path: str, df: pd.DataFrame | None = None) -> ErrorModel:
    """Загружает модель искажений из JSON-файла.

    df нужен только для селекторов групп ('*', '*numeric', ...): типы колонок
    определяются по фактической таблице.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return error_model_from_dict(data, df=df)

# ============================================================================
# 4. ГЛАВНЫЙ ИНЖЕКТОР
# ============================================================================

def _value_changed(old: Any, new: Any) -> bool:
    """True, если значение фактически изменилось (включая появление NaN)."""
    if pd.isna(old) != pd.isna(new):
        return True
    if pd.isna(old):
        return False
    try:
        return bool(old != new)
    except Exception:
        return str(old) != str(new)

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Корреляция Пирсона с защитой от вырожденных случаев (NaN/константа -> 0.0)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0

def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[np.argsort(values, kind="stable")] = np.arange(len(values), dtype=np.int64)
    return ranks

def _gaussian_scores(values: np.ndarray) -> np.ndarray:
    """Гауссовы scores эмпирического распределения: Ф^{-1}((rank - 0.5) / n)."""
    ranks = rankdata(values, method="average")
    u = (ranks - 0.5) / len(values)
    return stats.norm.ppf(u)

def _mixture_rewrite(y: np.ndarray, x: np.ndarray, target: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Прежний механизм: z-смесь col1 и перемешанной col2.

    Точно достигает целевой корреляции Пирсона и сохраняет mean/std col2,
    но меняет форму маргинального распределения col2 (кроме target=0,
    когда это фактически перестановка значений).
    """
    mu_y, sd_y = float(np.mean(y)), float(np.std(y, ddof=1)) or 1.0
    mu_x, sd_x = float(np.mean(x)), float(np.std(x, ddof=1)) or 1.0
    z_x = (x - mu_x) / sd_x
    z_y = (y - mu_y) / sd_y

    if abs(target) < 1e-9:
        y_new_z = z_y.copy()
        rng.shuffle(y_new_z)
    else:
        shuffled = z_y.copy()
        rng.shuffle(shuffled)
        coef = float(np.sqrt(max(0.0, 1.0 - target * target)))
        y_new_z = target * z_x + coef * shuffled
        mix_std = float(np.std(y_new_z, ddof=1))
        if mix_std > 1e-12:
            y_new_z = y_new_z / mix_std
    return mu_y + sd_y * y_new_z

def _permute_to_target(x: np.ndarray, y: np.ndarray, target: float,
                       rng: np.random.Generator, max_iter: int = 14,
                       tol: float = 2e-3,
                       ) -> tuple[np.ndarray, float, bool, tuple[float, float]]:
    """Строит y' - ПЕРЕСТАНОВКУ значений y, максимально близкую по корреляции
    Пирсона с x к целевой. Маргинальное распределение col2 сохраняется
    в точности (мультимножество значений не меняется).

    Механизм: гауссова связка над рангами x с коэффициентом r_eff;
    ранги связки задают перестановку отсортированного y. Коэффициент
    подбирается бисекцией по фактической корреляции Пирсона, поэтому целевая
    корреляция достигается с высокой точностью (tol), если она лежит в
    достижимых для перестановок границах [контрамонотонная, комонотонная];
    иначе target обрезается до ближайшей границы (clamped=True).

    Возвращает (y_new, achieved_pearson, clamped, (r_min, r_max)).
    """
    y_asc = np.sort(y)
    rx = _ordinal_ranks(x)
    r_max = _pearson(x, y_asc[rx])          # комонотонная перестановка (максимум)
    r_min = _pearson(x, y_asc[rx[::-1]])    # контрамонотонная (минимум)
    lo_bound, hi_bound = min(r_min, r_max), max(r_min, r_max)
    t = float(min(max(target, lo_bound), hi_bound))
    clamped = abs(t - target) > 1e-6

    z_x = _gaussian_scores(x)
    z_h = rng.standard_normal(len(x))

    def build(r_eff: float) -> np.ndarray:
        coef = float(np.sqrt(max(0.0, 1.0 - r_eff * r_eff)))
        u = stats.norm.cdf(r_eff * z_x + coef * z_h)
        return y_asc[_ordinal_ranks(u)]

    y0 = build(t)
    r0 = _pearson(x, y0)
    best_y, best_r, best_diff = y0, r0, abs(r0 - t)

    lo, hi = -1.0, 1.0
    for _ in range(max_iter):
        if best_diff < tol:
            break
        mid = 0.5 * (lo + hi)
        y_mid = build(mid)
        r_mid = _pearson(x, y_mid)
        if abs(r_mid - t) < best_diff:
            best_y, best_r, best_diff = y_mid, r_mid, abs(r_mid - t)
        if r_mid < t:
            lo = mid
        else:
            hi = mid
    return best_y, float(best_r), clamped, (float(lo_bound), float(hi_bound))

def inject_errors(
    df: pd.DataFrame,
    model: ErrorModel,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, DistortionReport]:
    py_rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    corrupted_df = df.copy()
    error_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    total_cells = len(df) * len(df.columns)
    planned_cells = 0
    corrupted_count = 0
    no_op_count = 0
    applied_marginal = []
    skipped: list[dict] = []
    warnings: list[str] = []

    enar_conditions: dict[str, np.ndarray] = {}
    for col, config in model.column_configs.items():
        if col not in df.columns:
            skipped.append({"column": col, "config": config.error_type, "reason": "column_not_found"})
            warnings.append(f"Колонка '{col}' из конфигурации отсутствует в данных - конфигурация пропущена")
            continue
        if config.mechanism == "enar":
            enar_conditions[col] = _eval_enar_condition(df, config.condition)

    # 1. Маргинальные искажения
    for col, config in model.column_configs.items():
        if col not in corrupted_df.columns:
            continue  

        series = corrupted_df[col]
        non_null_mask = series.notna()

        if config.error_type == "typo" and pd.api.types.is_datetime64_any_dtype(series):
            skipped.append({"column": col, "config": "typo", "reason": "typo_not_supported_for_datetime"})
            warnings.append(
                f"Колонка '{col}' имеет тип datetime64 - 'typo' для неё отключён "
                f"(давал непарсируемые строки). Используйте error_type='missing'."
            )
            continue

        if config.error_type in NUMERIC_ONLY_ERROR_TYPES and not (
            pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        ):
            skipped.append({"column": col, "config": config.error_type, "reason": "column_not_numeric"})
            warnings.append(
                f"Колонка '{col}' не числовая (dtype={series.dtype}) - "
                f"'{config.error_type}' неприменим, конфигурация пропущена"
            )
            continue

        if config.error_type == "precision_loss" and pd.api.types.is_integer_dtype(series):
            skipped.append({"column": col, "config": "precision_loss", "reason": "integer_column"})
            warnings.append(
                f"Колонка '{col}' целочисленная - precision_loss не изменил бы значений, "
                f"конфигурация пропущена"
            )
            continue

        valid_cats = None
        if config.error_type == "category_swap":
            valid_cats = series.dropna().unique().tolist()
            if len(valid_cats) <= 1:
                skipped.append({"column": col, "config": "category_swap", "reason": "fewer_than_two_categories"})
                warnings.append(
                    f"Колонка '{col}': меньше двух категорий - category_swap неприменим, конфигурация пропущена"
                )
                continue

        if config.error_type in ("noise_additive", "noise_multiplicative") and pd.api.types.is_integer_dtype(series):
            corrupted_df[col] = corrupted_df[col].astype("float64")
            series = corrupted_df[col]
            warnings.append(f"Колонка '{col}' приведена к float64 для шумового искажения")

        if config.error_type in ("outlier", "numeric_typo", "missing") and pd.api.types.is_integer_dtype(series):
            corrupted_df[col] = corrupted_df[col].astype("float64")
            series = corrupted_df[col]
            warnings.append(f"Колонка '{col}' приведена к float64 для '{config.error_type}'")
        if config.error_type == "missing" and pd.api.types.is_bool_dtype(series):
            corrupted_df[col] = corrupted_df[col].astype("object")
            series = corrupted_df[col]
            warnings.append(f"Колонка '{col}' приведена к object для 'missing' (bool не хранит NaN)")

        if config.mechanism == "enar":
            mask = (np_rng.random(len(series)) < config.error_rate) & enar_conditions[col] & non_null_mask.to_numpy()
        else:
            mask = get_ecar_mask(len(series), config.error_rate, np_rng) & non_null_mask.values

        planned = int(mask.sum())
        planned_cells += planned
        error_mask.loc[mask, col] = True

        changed = 0
        if config.error_type in VECTORIZED_ERROR_TYPES:
            values = series.to_numpy(dtype=float)
            selected = values[mask]
            if config.error_type == "noise_additive":
                rel_sigma = config.intensity if config.intensity is not None else 0.1
                col_std = float(np.nanstd(values))
                noise = np_rng.normal(0.0, max(col_std, 1e-12) * rel_sigma, size=selected.size)
                new_values = selected + noise
            elif config.error_type == "noise_multiplicative":
                rel_sigma = config.intensity if config.intensity is not None else 0.1
                new_values = selected * (1.0 + np_rng.normal(0.0, rel_sigma, size=selected.size))
            else:  # precision_loss
                decimals = int(config.intensity) if config.intensity is not None else 1
                new_values = np.round(selected, decimals)
            corrupted_df.loc[mask, col] = new_values
            changed = int(np.sum(new_values != selected))
        else:
            error_func = ERROR_TYPE_FUNCS[config.error_type]
            for idx in series[mask].index:
                val = series.loc[idx]
                new_val = error_func(val, py_rng, valid_cats) if valid_cats is not None else error_func(val, py_rng)
                corrupted_df.loc[idx, col] = new_val
                changed += int(_value_changed(val, new_val))

        corrupted_count += changed
        no_op_count += planned - changed
        if planned > 0 and changed == 0:
            warnings.append(f"Колонка '{col}': выбрано {planned} ячеек, но ни одно значение фактически не изменилось")

        entry = {
            "column": col,
            "type": config.error_type,
            "rate": config.error_rate,
            "mechanism": config.mechanism,
            "planned": planned,
            "corrupted": changed,
        }
        if config.matched_by:
            entry["matched_by"] = config.matched_by
        applied_marginal.append(entry)

    # 2. Парные искажения 
    applied_pairwise = []

    for p_cfg in model.pairwise_configs:
        missing_cols = [c for c in (p_cfg.col1, p_cfg.col2) if c not in corrupted_df.columns]
        if missing_cols:
            skipped.append({"column": ",".join(missing_cols), "config": p_cfg.action, "reason": "column_not_found"})
            warnings.append(f"Парное искажение {p_cfg.action} пропущено: нет колонок {missing_cols}")
            continue

        try:
            orig_corr = float(corrupted_df[[p_cfg.col1, p_cfg.col2]].corr().iloc[0, 1])
        except Exception:
            orig_corr = None

        if p_cfg.action in ("break_correlation", "false_link"):
            target = float(p_cfg.target_correlation)
            x = corrupted_df[p_cfg.col1]
            y = corrupted_df[p_cfg.col2]

            if not (pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
                    and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)):
                skipped.append({
                    "column": f"{p_cfg.col1},{p_cfg.col2}",
                    "config": p_cfg.action,
                    "reason": "columns_not_numeric",
                })
                warnings.append(
                    f"{p_cfg.action}({p_cfg.col1}, {p_cfg.col2}): обе колонки должны быть "
                    f"числовыми (не bool) - пропущено"
                )
                continue

            x_clean = x.astype("float64")
            y_clean = y.astype("float64")
            valid = x_clean.notna() & y_clean.notna()

            if valid.sum() < 3:
                skipped.append({
                    "column": f"{p_cfg.col1},{p_cfg.col2}",
                    "config": p_cfg.action,
                    "reason": "too_few_complete_rows",
                })
                warnings.append(
                    f"{p_cfg.action}({p_cfg.col1}, {p_cfg.col2}): слишком мало полных строк - пропущено"
                )
                continue

            x_vals = x_clean[valid].to_numpy(dtype="float64")
            y_vals = y_clean[valid].to_numpy(dtype="float64")

            clamped = False
            bounds: tuple[float, float] | None = None
            if p_cfg.method == "mixture":
                y_new = _mixture_rewrite(y_vals, x_vals, target, np_rng)
                marginal_preserved = False
            else:
                y_new, r_fact, clamped, bounds = _permute_to_target(
                    x_vals, y_vals, target, np_rng
                )
                marginal_preserved = True
                if clamped:
                    warnings.append(
                        f"{p_cfg.action}({p_cfg.col1}, {p_cfg.col2}): target_correlation="
                        f"{target} недостижим перестановкой (границы [{bounds[0]:.3f}, "
                        f"{bounds[1]:.3f}]) при сохранении маргинального распределения "
                        f"'{p_cfg.col2}' - применена ближайшая достижимая корреляция "
                        f"{r_fact:.3f}"
                    )

            # Приведение типа выполняется только перед фактической записью значений:
            # если бы искажение было пропущено выше, dtype колонки остался бы неизменным.
            if pd.api.types.is_integer_dtype(corrupted_df[p_cfg.col2]):
                corrupted_df[p_cfg.col2] = corrupted_df[p_cfg.col2].astype("float64")
                warnings.append(
                    f"Колонка '{p_cfg.col2}' приведена к float64 для '{p_cfg.action}'"
                )

            corrupted_df.loc[valid[valid].index, p_cfg.col2] = np.asarray(y_new, dtype="float64")

            try:
                new_corr = float(corrupted_df[[p_cfg.col1, p_cfg.col2]].corr().iloc[0, 1])
            except Exception:
                new_corr = None

            applied_pairwise.append({
                "action": p_cfg.action,
                "col1": p_cfg.col1,
                "col2": p_cfg.col2,
                "original_correlation": orig_corr,
                "new_correlation": new_corr,
                "target_correlation": p_cfg.target_correlation,
                "method": p_cfg.method,
                "marginal_preserved": marginal_preserved,
            })
        elif p_cfg.action == "shift_dependency":
            x = corrupted_df[p_cfg.col1]
            y = corrupted_df[p_cfg.col2]

            if not (pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
                    and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)):
                skipped.append({
                    "column": f"{p_cfg.col1},{p_cfg.col2}",
                    "config": "shift_dependency",
                    "reason": "columns_not_numeric",
                })
                warnings.append(
                    f"shift_dependency({p_cfg.col1}, {p_cfg.col2}): обе колонки должны быть "
                    f"числовыми (не bool) - пропущено"
                )
                continue

            x_clean = x.astype("float64")
            y_clean = y.astype("float64")
            valid = x_clean.notna() & y_clean.notna()

            if valid.sum() < 3:
                skipped.append({
                    "column": f"{p_cfg.col1},{p_cfg.col2}",
                    "config": "shift_dependency",
                    "reason": "too_few_complete_rows",
                })
                warnings.append(
                    f"shift_dependency({p_cfg.col1}, {p_cfg.col2}): слишком мало полных строк - пропущено"
                )
                continue

            xv = x_clean[valid].to_numpy(dtype="float64")
            yv = y_clean[valid].to_numpy(dtype="float64")

            sd_x = float(xv.std(ddof=1)) or 1.0
            sd_y = float(yv.std(ddof=1)) or 1.0
            r_target = float(p_cfg.target_slope) * sd_x / sd_y

            a_orig = float(np.cov(xv, yv, ddof=1)[0, 1] / (np.var(xv, ddof=1) or 1e-12))

            if p_cfg.method == "mixture":
                if abs(r_target) > 1.0:
                    skipped.append({
                        "column": f"{p_cfg.col1},{p_cfg.col2}",
                        "config": "shift_dependency",
                        "reason": f"target_slope_unreachable (implied |r|={abs(r_target):.3f} > 1)",
                    })
                    warnings.append(
                        f"shift_dependency({p_cfg.col1}, {p_cfg.col2}): target_slope="
                        f"{p_cfg.target_slope} требует |corr|={abs(r_target):.3f} > 1, "
                        f"что недостижимо при сохранении std(col2) - пропущено"
                    )
                    continue
                y_new = _mixture_rewrite(yv, xv, r_target, np_rng)
                marginal_preserved = False
            else:
                y_new, r_fact, clamped, bounds = _permute_to_target(
                    xv, yv, r_target, np_rng
                )
                marginal_preserved = True
                if clamped:
                    warnings.append(
                        f"shift_dependency({p_cfg.col1}, {p_cfg.col2}): target_slope="
                        f"{p_cfg.target_slope} требует корреляцию {r_target:.3f}, "
                        f"недостижимую перестановкой (границы [{bounds[0]:.3f}, "
                        f"{bounds[1]:.3f}]) при сохранении маргинального распределения "
                        f"'{p_cfg.col2}' - применён ближайший достижимый наклон"
                    )

            # Приведение типа выполняется только перед фактической записью значений:
            # если бы искажение было пропущено выше, dtype колонки остался бы неизменным.
            if pd.api.types.is_integer_dtype(corrupted_df[p_cfg.col2]):
                corrupted_df[p_cfg.col2] = corrupted_df[p_cfg.col2].astype("float64")
                warnings.append(
                    f"Колонка '{p_cfg.col2}' приведена к float64 для '{p_cfg.action}'"
                )

            corrupted_df.loc[valid[valid].index, p_cfg.col2] = np.asarray(y_new, dtype="float64")

            try:
                a_fact = float(np.cov(xv, y_new, ddof=1)[0, 1] / (np.var(xv, ddof=1) or 1e-12))
            except Exception:
                a_fact = None

            applied_pairwise.append({
                "action": "shift_dependency",
                "col1": p_cfg.col1,
                "col2": p_cfg.col2,
                "original_slope": a_orig,
                "new_slope": a_fact,
                "target_slope": p_cfg.target_slope,
                "original_correlation": orig_corr,
                "method": p_cfg.method,
                "marginal_preserved": marginal_preserved,
            })
        elif p_cfg.action == "invert_correlation":
            if pd.api.types.is_numeric_dtype(corrupted_df[p_cfg.col2]) \
                    and not pd.api.types.is_bool_dtype(corrupted_df[p_cfg.col2]):
                mean_val = corrupted_df[p_cfg.col2].mean()
                corrupted_df[p_cfg.col2] = mean_val - (corrupted_df[p_cfg.col2] - mean_val)

                try:
                    new_corr = float(corrupted_df[[p_cfg.col1, p_cfg.col2]].corr().iloc[0, 1])
                except Exception:
                    new_corr = None

                applied_pairwise.append({
                    "action": "invert_correlation",
                    "col1": p_cfg.col1,
                    "col2": p_cfg.col2,
                    "original_correlation": orig_corr,
                    "new_correlation": new_corr,
                })
            else:
                skipped.append({"column": p_cfg.col2, "config": "invert_correlation", "reason": "column_not_numeric"})
                warnings.append(
                    f"invert_correlation({p_cfg.col1}, {p_cfg.col2}): колонка '{p_cfg.col2}' "
                    f"не числовая - пропущено"
                )

    # 3. Структурные искажения 
    applied_structural: list[dict] = []

    for s_cfg in model.structural_configs:
        if s_cfg.action == "duplicate_rows":
            n_dup = int(round(len(corrupted_df) * float(s_cfg.rate)))
            if n_dup <= 0:
                warnings.append(f"duplicate_rows: rate={s_cfg.rate} не дал ни одной дублируемой строки")
                continue
            source_pos = np_rng.choice(len(corrupted_df), size=n_dup, replace=False)
            dup_rows = corrupted_df.iloc[source_pos]
            dup_mask = error_mask.iloc[source_pos]
            if isinstance(corrupted_df.index, pd.RangeIndex):
                corrupted_df = pd.concat([corrupted_df, dup_rows], ignore_index=True)
                error_mask = pd.concat([error_mask, dup_mask], ignore_index=True)
            else:
                corrupted_df = pd.concat([corrupted_df, dup_rows])
                error_mask = pd.concat([error_mask, dup_mask])
                warnings.append(
                    "Индекс таблицы не RangeIndex: дублированные строки получили копии исходных меток индекса"
                )
            applied_structural.append({"action": "duplicate_rows", "rows_added": n_dup, "rate": s_cfg.rate})

        elif s_cfg.action == "drop_column":
            for c in s_cfg.columns:
                if c in corrupted_df.columns:
                    corrupted_df = corrupted_df.drop(columns=[c])
                    error_mask = error_mask.drop(columns=[c])
                    applied_structural.append({"action": "drop_column", "column": c})
                else:
                    skipped.append({"column": c, "config": "drop_column", "reason": "column_not_found"})
                    warnings.append(f"drop_column: колонка '{c}' отсутствует - пропущено")

        elif s_cfg.action == "rename_column":
            c = s_cfg.columns[0]
            if c not in corrupted_df.columns:
                skipped.append({"column": c, "config": "rename_column", "reason": "column_not_found"})
                warnings.append(f"rename_column: колонка '{c}' отсутствует - пропущено")
            elif s_cfg.new_name in corrupted_df.columns:
                skipped.append({"column": c, "config": "rename_column", "reason": "target_name_exists"})
                warnings.append(f"rename_column: колонка '{s_cfg.new_name}' уже существует - пропущено")
            else:
                corrupted_df = corrupted_df.rename(columns={c: s_cfg.new_name})
                error_mask = error_mask.rename(columns={c: s_cfg.new_name})
                applied_structural.append({"action": "rename_column", "column": c, "new_name": s_cfg.new_name})

        elif s_cfg.action == "change_type":
            c = s_cfg.columns[0]
            if c not in corrupted_df.columns:
                skipped.append({"column": c, "config": "change_type", "reason": "column_not_found"})
                warnings.append(f"change_type: колонка '{c}' отсутствует - пропущено")
                continue
            before_dtype = str(corrupted_df[c].dtype)
            try:
                if s_cfg.new_type == "float":
                    corrupted_df[c] = pd.to_numeric(corrupted_df[c], errors="coerce").astype("float64")
                elif s_cfg.new_type == "int":
                    corrupted_df[c] = pd.to_numeric(corrupted_df[c], errors="coerce").astype("Int64")
                elif s_cfg.new_type == "string":
                    corrupted_df[c] = corrupted_df[c].astype(str)
                elif s_cfg.new_type == "datetime":
                    parsed = pd.to_datetime(corrupted_df[c], errors="coerce")
                    newly_broken = int((parsed.isna() & corrupted_df[c].notna()).sum())
                    corrupted_df[c] = parsed
                    if newly_broken:
                        warnings.append(
                            f"change_type '{c}' -> datetime: {newly_broken} значений не распознаны и стали NaT"
                        )
                applied_structural.append({
                    "action": "change_type",
                    "column": c,
                    "from_dtype": before_dtype,
                    "to_dtype": str(corrupted_df[c].dtype),
                })
            except Exception as e:
                skipped.append({"column": c, "config": "change_type", "reason": f"cast_failed: {e}"})
                warnings.append(f"change_type для '{c}' не выполнен: {e}")

        elif s_cfg.action == "distribution_drift":
            c = s_cfg.columns[0]
            if c not in corrupted_df.columns:
                skipped.append({"column": c, "config": "distribution_drift", "reason": "column_not_found"})
                warnings.append(f"distribution_drift: колонка '{c}' отсутствует - пропущено")
                continue
            series = corrupted_df[c]
            if not (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)):
                skipped.append({"column": c, "config": "distribution_drift", "reason": "column_not_numeric"})
                warnings.append(
                    f"distribution_drift: колонка '{c}' не числовая (dtype={series.dtype}) - пропущено"
                )
                continue
            before_mean = float(series.dropna().mean()) if series.notna().any() else None
            before_std = float(series.dropna().std(ddof=1)) if series.notna().sum() > 1 else None
            if pd.api.types.is_integer_dtype(series):
                corrupted_df[c] = series.astype("float64")
                series = corrupted_df[c]
                warnings.append(f"distribution_drift: колонка '{c}' приведена к float64")
            delta = float(s_cfg.delta)
            scale = float(s_cfg.scale)
            new_series = (series.astype("float64") + delta) * scale
            corrupted_df[c] = new_series
            after_mean = float(new_series.dropna().mean()) if new_series.notna().any() else None
            after_std = float(new_series.dropna().std(ddof=1)) if new_series.notna().sum() > 1 else None
            applied_structural.append({
                "action": "distribution_drift",
                "column": c,
                "delta": delta,
                "scale": scale,
                "before_mean": before_mean,
                "after_mean": after_mean,
                "before_std": before_std,
                "after_std": after_std,
            })

        elif s_cfg.action == "violate_constraints":
            for c in s_cfg.columns:
                if c not in corrupted_df.columns:
                    skipped.append({"column": c, "config": "violate_constraints", "reason": "column_not_found"})
                    warnings.append(f"violate_constraints: колонка '{c}' отсутствует - пропущено")
                    continue

                series = corrupted_df[c]
                n_rows = len(corrupted_df)
                n_target = int(round(n_rows * float(s_cfg.rate)))
                if n_target <= 0:
                    warnings.append(
                        f"violate_constraints ({c}, {s_cfg.constraint}): "
                        f"rate={s_cfg.rate} не дал ни одной строки - пропущено"
                    )
                    continue

                if s_cfg.constraint == "unique":
                    non_null_idx = series.dropna().index
                    if len(non_null_idx) < 2:
                        skipped.append({"column": c, "config": "violate_constraints", "reason": "not_enough_values"})
                        warnings.append(
                            f"violate_constraints (unique): в колонке '{c}' меньше двух заполненных значений - пропущено"
                        )
                        continue
                    n_target = min(n_target, len(non_null_idx))
                    pos = np_rng.choice(len(non_null_idx), size=n_target, replace=False)
                    target_idx = non_null_idx[pos]

                    donor_pos = (pos + np_rng.integers(1, len(non_null_idx), size=n_target)) % len(non_null_idx)
                    donor_idx = non_null_idx[donor_pos]
                    donor_vals = corrupted_df.loc[donor_idx, c].to_numpy()
                    unique_before = int(series.dropna().nunique())
                    changed = 0
                    for idx, val in zip(target_idx, donor_vals):
                        old = corrupted_df.at[idx, c]
                        corrupted_df.at[idx, c] = val
                        changed += int(_value_changed(old, val))
                    error_mask.loc[target_idx, c] = True
                    unique_after = int(corrupted_df[c].dropna().nunique())
                    n_valid_after = int(corrupted_df[c].notna().sum())
                    applied_structural.append({
                        "action": "violate_constraints",
                        "column": c,
                        "constraint": "unique",
                        "rate": s_cfg.rate,
                        "planned": n_target,
                        "corrupted": changed,
                        "unique_before": unique_before,
                        "unique_after": unique_after,
                        "duplicates_added": max(0, (n_valid_after - unique_after) - (n_valid_after - unique_before)),
                    })

                elif s_cfg.constraint == "range":
                    if not (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)):
                        skipped.append({"column": c, "config": "violate_constraints", "reason": "column_not_numeric"})
                        warnings.append(
                            f"violate_constraints (range): колонка '{c}' не числовая (dtype={series.dtype}) - пропущено"
                        )
                        continue
                    lo = None if s_cfg.min_value is None else float(s_cfg.min_value)
                    hi = None if s_cfg.max_value is None else float(s_cfg.max_value)
                    non_null_idx = series.dropna().index
                    if len(non_null_idx) == 0:
                        skipped.append({"column": c, "config": "violate_constraints", "reason": "no_values"})
                        warnings.append(f"violate_constraints (range): колонка '{c}' пуста - пропущено")
                        continue
                    n_target = min(n_target, len(non_null_idx))
                    # Относительный офсет, чтобы нарушения были заметны на любом масштабе данных
                    if lo is not None and hi is not None:
                        span = hi - lo
                        base = span if span > 0 else max(abs(lo), abs(hi), 1.0)
                    elif lo is not None:
                        base = max(abs(lo), 1.0)
                    else:
                        base = max(abs(hi), 1.0)
                    offset = 0.1 * base

                    def _violation_value() -> float:
                        if lo is not None and hi is not None and np_rng.random() < 0.5:
                            return lo - np_rng.uniform(0.2, 1.0) * offset
                        if hi is not None:
                            return hi + np_rng.uniform(0.2, 1.0) * offset
                        return lo - np_rng.uniform(0.2, 1.0) * offset

                    is_int = pd.api.types.is_integer_dtype(series)
                    pos = np_rng.choice(len(non_null_idx), size=n_target, replace=False)
                    target_idx = non_null_idx[pos]
                    changed = 0
                    for idx in target_idx:
                        old = corrupted_df.at[idx, c]
                        v = _violation_value()
                        if is_int:
                            v = int(round(v))
                            corrupted_df.at[idx, c] = v
                        else:
                            corrupted_df.at[idx, c] = float(v)
                        changed += int(_value_changed(old, v))
                    error_mask.loc[target_idx, c] = True
                    after = corrupted_df[c]
                    oob_mask = pd.Series(False, index=after.index)
                    if lo is not None:
                        oob_mask |= after < lo
                    if hi is not None:
                        oob_mask |= after > hi
                    applied_structural.append({
                        "action": "violate_constraints",
                        "column": c,
                        "constraint": "range",
                        "rate": s_cfg.rate,
                        "planned": n_target,
                        "corrupted": changed,
                        "min_value": lo,
                        "max_value": hi,
                        "out_of_bounds_after": int(oob_mask.sum()),
                    })

                elif s_cfg.constraint == "not_null":
                    if pd.api.types.is_integer_dtype(series):
                        corrupted_df[c] = series.astype("float64")
                        series = corrupted_df[c]
                        warnings.append(f"violate_constraints (not_null): колонка '{c}' приведена к float64")
                    elif pd.api.types.is_bool_dtype(series):
                        corrupted_df[c] = series.astype("object")
                        series = corrupted_df[c]
                        warnings.append(f"violate_constraints (not_null): колонка '{c}' приведена к object")
                    non_null_idx = series.dropna().index
                    if len(non_null_idx) == 0:
                        skipped.append({"column": c, "config": "violate_constraints", "reason": "no_values"})
                        warnings.append(f"violate_constraints (not_null): колонка '{c}' пуста - пропущено")
                        continue
                    n_target = min(n_target, len(non_null_idx))
                    pos = np_rng.choice(len(non_null_idx), size=n_target, replace=False)
                    target_idx = non_null_idx[pos]
                    corrupted_df.loc[target_idx, c] = np.nan
                    error_mask.loc[target_idx, c] = True
                    applied_structural.append({
                        "action": "violate_constraints",
                        "column": c,
                        "constraint": "not_null",
                        "rate": s_cfg.rate,
                        "planned": n_target,
                        "corrupted": n_target,
                        "new_nulls": n_target,
                    })

                else:
                    skipped.append({
                        "column": c,
                        "config": "violate_constraints",
                        "reason": f"unknown_constraint: {s_cfg.constraint}",
                    })
                    warnings.append(f"violate_constraints: неизвестное ограничение '{s_cfg.constraint}' - пропущено")

    structural_changes = {"applied": applied_structural} if applied_structural else {}

    if not applied_marginal and not applied_pairwise and not applied_structural:
        warnings.append(
            "Ни одна конфигурация искажений не была применена - проверьте поля skipped и warnings отчёта"
        )

    actual_error_rate = corrupted_count / total_cells if total_cells > 0 else 0.0

    report = DistortionReport(
        target_error_rate=sum(c.error_rate for c in model.column_configs.values()) / max(1, len(model.column_configs)),
        actual_error_rate=float(actual_error_rate),
        corrupted_cells_count=corrupted_count,
        applied_marginal_models=applied_marginal,
        applied_pairwise_actions=applied_pairwise,
        structural_changes=structural_changes,
        planned_cells_count=planned_cells,
        no_op_cells_count=no_op_count,
        warnings=warnings,
        skipped=skipped
    )

    return corrupted_df, error_mask, report


# ============================================================================
# 5. АВТОГЕНЕРАЦИЯ МОДЕЛИ
# ============================================================================

def auto_error_model_from_profile(
    profile: TableProfile,
    error_rate: float = 0.1,
    exclude_columns: list[str] | None = None,
) -> ErrorModel:
    """Автоматическая модель искажений на основе профиля данных.

    Маргинальные: numeric - выбросы, categorical/boolean - смена категории,
    text - опечатки, datetime - пропуски.
    Структурные: identifier - нарушение
    уникальности (unique), numeric - выход за наблюдаемый диапазон
    (range с границами min/max из профиля).
    """
    exclude_columns = exclude_columns or []
    configs = {}
    structural_configs: list[StructuralDistortionConfig] = []

    for col_profile in profile.columns:
        col_name = col_profile.name
        if col_name in exclude_columns or col_profile.missing_fraction > 0.5:
            continue

        kind = col_profile.kind
        if kind == "numeric":
            configs[col_name] = ColumnErrorConfig(error_type="outlier", error_rate=error_rate, mechanism="ecar")
        elif kind == "categorical" and len(col_profile.categories) > 1:
            configs[col_name] = ColumnErrorConfig(error_type="category_swap", error_rate=error_rate, mechanism="ecar")
        elif kind == "boolean" and len(col_profile.bool_values) > 1:
            configs[col_name] = ColumnErrorConfig(error_type="category_swap", error_rate=error_rate, mechanism="ecar")
        elif kind == "text":
            configs[col_name] = ColumnErrorConfig(error_type="typo", error_rate=error_rate, mechanism="ecar")
        elif kind == "datetime":
            configs[col_name] = ColumnErrorConfig(error_type="missing", error_rate=error_rate, mechanism="ecar")

        if kind == "identifier" and col_profile.id_is_integer:
            structural_configs.append(StructuralDistortionConfig(
                action="violate_constraints", columns=[col_name], constraint="unique", rate=error_rate,
            ))
        elif kind == "numeric" and col_profile.min is not None and col_profile.max is not None:
            structural_configs.append(StructuralDistortionConfig(
                action="violate_constraints", columns=[col_name], constraint="range", rate=error_rate,
                min_value=float(col_profile.min), max_value=float(col_profile.max),
            ))

    return ErrorModel(column_configs=configs, structural_configs=structural_configs)

def save_error_model(model: ErrorModel, json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2, ensure_ascii=False)
