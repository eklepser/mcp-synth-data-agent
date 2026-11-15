# src/synth_service/error_injector.py
#
# Модуль искажений. Изменения (патч "пункты 1-3" по результатам ревизии):
#   1. ENAR-условия вычисляются на ПОЛНОМ фрейме (можно ссылаться на любые
#      колонки) и по ИСХОДНЫМ данным (результат не зависит от порядка
#      конфигураций); валидация конфигураций (error_rate в [0,1], известные
#      типы/механизмы, enar требует condition); любые пропуски конфигураций
#      фиксируются в отчёте (warnings/skipped) вместо молчаливых.
#   2. numeric_typo больше не роняет float() (дублируются только цифровые
#      символы; варианты без эффекта заменяются на shift); typo на
#      datetime64-колонках отключён (в auto-модели datetime -> missing).
#   3. Каталог расширен до тезисов: шум (noise_additive / noise_multiplicative),
#      потеря точности (precision_loss) и структурные искажения
#      (duplicate_rows, drop_column, rename_column, change_type).
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd

from synth_service.profile import TableProfile

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
    """Опечатка в строковом представлении числа.

    "duplicate_digit" дублирует только цифровой символ: раньше задвоение '.'
    или '-' давало float(...) -> ValueError и роняло весь конвейер.
    Варианты, которые фактически не меняют значение (нет нуля в строке,
    задвоение хвостового нуля и т.п.), заменяются на "shift", чтобы
    фактический уровень искажений соответствовал заявленному.
    """
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
    # duplicate_digit
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

# Типы, обрабатываемые векторно (получают сразу массив выбранных значений):
#  - noise_additive / noise_multiplicative: модели шума из тезисов
#  - precision_loss: потеря точности (округление) из тезисов
VECTORIZED_ERROR_TYPES = ("noise_additive", "noise_multiplicative", "precision_loss")

MARGINAL_ERROR_TYPES = set(ERROR_TYPE_FUNCS) | set(VECTORIZED_ERROR_TYPES)

# Типы, применимые только к числовым (не bool) колонкам
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
    """Вычисляет ENAR-условие на полном фрейме.

    Раньше условие оценивалось на фрейме из одной колонки, поэтому ссылка на
    другую колонку падала с ValueError - а зависимость от значений других
    столбцов и есть суть ENAR.
    """
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
    """Конфигурация маргинального искажения одной колонки.

    error_rate - доля ячеек (маскируемых по ecar/enar), которые планируется
                 исказить; строго в [0, 1] (раньше любые значения проходили молча).
    intensity  - сила искажения помимо доли ячеек (ТЕЗИС: "типа и интенсивности"):
                 noise_additive       - sigma шума как доля std колонки (по умолчанию 0.1);
                 noise_multiplicative - sigma относительного шума (по умолчанию 0.1);
                 precision_loss       - сколько знаков после запятой оставить (по умолчанию 1).
    """
    error_type: Literal["typo", "numeric_typo", "outlier", "missing", "category_swap",
                        "noise_additive", "noise_multiplicative", "precision_loss"]
    error_rate: float
    mechanism: Literal["ecar", "enar"] = "ecar"
    condition: str | None = None
    intensity: float | None = None

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
    """Конфигурация для парных искажений (ТЕЗИС: разрыв, ослабление, ложная связь, инверсия, сдвиг).

    break_correlation   - ослабление/разрыв корреляции col1<->col2 до target_correlation
                          (по умолчанию 0.0 = полный разрыв через перестановку col2).
                          Если target_correlation != 0, исходный столбец смешивается с
                          переставленным в пропорции, обеспечивающей целевую корреляцию.
    invert_correlation  - инверсия знака зависимости.
    shift_dependency    - изменение наклона линейной зависимости col2 = a*col1 + b + eps
                          до target_slope (сохраняя маргинальное среднее/std col2).
    """
    col1: str
    col2: str
    action: Literal["break_correlation", "invert_correlation", "shift_dependency"]
    target_correlation: float = 0.0  # Для break_correlation: целевая корреляция в [-1, 1]
    target_slope: float | None = None  # Для shift_dependency: целевой коэффициент при col1

    def __post_init__(self):
        if self.action not in ("break_correlation", "invert_correlation", "shift_dependency"):
            raise ValueError(
                f"Неизвестное парное действие: {self.action!r}. "
                f"Доступны: break_correlation, invert_correlation, shift_dependency"
            )
        if self.action == "break_correlation" and not (-1.0 <= float(self.target_correlation) <= 1.0):
            raise ValueError(
                f"target_correlation должен быть в [-1, 1], получено: {self.target_correlation!r}"
            )
        if self.action == "shift_dependency" and self.target_slope is None:
            raise ValueError("shift_dependency требует target_slope (числовой коэффициент наклона)")

STRUCTURAL_ACTIONS = ("duplicate_rows", "drop_column", "rename_column", "change_type", "distribution_drift")
CHANGE_TYPE_TARGETS = ("float", "int", "string", "datetime")

@dataclass
class StructuralDistortionConfig:
    """Структурное искажение (ТЕЗИС: класс структурных - таблица в целом).

    duplicate_rows      - дублирование rate-доли строк;
    drop_column         - удаление колонок из columns;
    rename_column       - переименование columns[0] в new_name;
    change_type         - смена типа columns[0] на new_type (float/int/string/datetime);
    distribution_drift  - дрейф распределения числовой колонки columns[0]:
                          Y' = (Y + delta) * scale
                          (delta - аддитивный сдвиг, по умолчанию 0;
                           scale - мультипликативный масштаб, по умолчанию 1).
                          Соответствует тезису "дрейф распределений" - сдвиг маргинала
                          всей колонки при сохранении формы.
    """
    action: Literal["duplicate_rows", "drop_column", "rename_column", "change_type", "distribution_drift"]
    columns: list[str] = field(default_factory=list)
    new_name: str | None = None
    new_type: str | None = None
    rate: float = 0.05
    delta: float = 0.0   # Для distribution_drift: аддитивный сдвиг
    scale: float = 1.0   # Для distribution_drift: мультипликативный масштаб

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

@dataclass
class DistortionReport:
    """Структурированный отчет о примененных искажениях (ТЕЗИС: возвращая структурированный отчёт)

    planned_cells_count   - сколько ячеек выбрано масками (план);
    corrupted_cells_count - сколько ячеек фактически изменено (функция искажения
                            могла не изменить выбранное значение);
    no_op_cells_count     - выбраны маской, но фактически не изменились
                            (раньше это расхождение было невидимым);
    warnings              - проблемы конфигурации/данных, не остановившие выполнение;
    skipped               - пропущенные конфигурации с причиной (раньше пропуски
                            были молчаливыми).
    """
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
    # Совместимость со старым форматом MCP-тула: "rate" -> "error_rate"
    if "rate" in cfg and "error_rate" not in cfg:
        cfg["error_rate"] = cfg.pop("rate")
    unknown = set(cfg) - set(ColumnErrorConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Конфигурация колонки '{col}' содержит неизвестные поля: {sorted(unknown)}")
    try:
        return ColumnErrorConfig(**cfg)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Конфигурация колонки '{col}': {e}") from e

def error_model_from_dict(data: dict) -> ErrorModel:
    """Строит ErrorModel из словаря конфигурации (тот же формат, что читает load_error_model).

    Поддерживаются формат {"marginal": {...}, "pairwise": [...], "structural": [...]}
    и плоский {"колонка": {...}} (раньше плоский формат молча давал 0 искажений).
    """
    if not isinstance(data, dict):
        raise ValueError("Конфигурация ошибок должна быть JSON-объектом")

    if "marginal" in data:
        marginal_data = data.get("marginal") or {}
    elif "pairwise" in data or "structural" in data:
        marginal_data = {}
    else:
        marginal_data = data

    configs = {}
    for col, cfg_dict in marginal_data.items():
        configs[col] = _build_column_config(col, cfg_dict)

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

def load_error_model(json_path: str) -> ErrorModel:
    """Загружает модель ошибок из JSON-файла (см. error_model_from_dict)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return error_model_from_dict(data)

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

    # Предварительный проход: молчаливые пропуски -> skipped/warnings,
    # ENAR-условия вычисляются на ИСХОДНОМ фрейме (до любых изменений), чтобы
    # результат не зависел от порядка конфигураций в модели.
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
            continue  # уже зафиксировано в skipped/warnings на предварительном проходе

        series = corrupted_df[col]
        non_null_mask = series.notna()

        # typo неприменим к datetime64: подмена символов даёт непарсируемые строки
        # и молча ломает тип колонки
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

        # Шум меняет дробные значения; целочисленную колонку приводим к float64 явно
        if config.error_type in ("noise_additive", "noise_multiplicative") and pd.api.types.is_integer_dtype(series):
            corrupted_df[col] = corrupted_df[col].astype("float64")
            series = corrupted_df[col]
            warnings.append(f"Колонка '{col}' приведена к float64 для шумового искажения")

        # Маска ячеек
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

        applied_marginal.append({
            "column": col,
            "type": config.error_type,
            "rate": config.error_rate,
            "mechanism": config.mechanism,
            "planned": planned,
            "corrupted": changed,
        })

    # 2. Парные искажения (ТЕЗИС: разрыв корреляции и др.)
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

        if p_cfg.action == "break_correlation":
            # ТЕЗИС: "ослабление или разрыв корреляционной связи".
            # target_correlation == 0: чистая перестановка col2 -> корреляция ~0.
            # target_correlation != 0: строим Y_new = target*r_z(x) + sqrt(1-target^2)*Y_shuffled_z
            # (стандартизуем X и Y, смешиваем с нужным весом, денормализуем обратно к mean/std Y).
            # Сохраняет маргинальное mean/std col2, обеспечивает целевую корреляцию с col1.
            target = float(p_cfg.target_correlation)
            x = corrupted_df[p_cfg.col1]
            y = corrupted_df[p_cfg.col2]

            if not (pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
                    and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)):
                skipped.append({
                    "column": f"{p_cfg.col1},{p_cfg.col2}",
                    "config": "break_correlation",
                    "reason": "columns_not_numeric",
                })
                warnings.append(
                    f"break_correlation({p_cfg.col1}, {p_cfg.col2}): обе колонки должны быть "
                    f"числовыми (не bool) - пропущено"
                )
                continue

            x_clean = x.astype("float64")
            y_clean = y.astype("float64")
            valid = x_clean.notna() & y_clean.notna()

            if valid.sum() < 3:
                skipped.append({
                    "column": f"{p_cfg.col1},{p_cfg.col2}",
                    "config": "break_correlation",
                    "reason": "too_few_complete_rows",
                })
                warnings.append(
                    f"break_correlation({p_cfg.col1}, {p_cfg.col2}): слишком мало полных строк - пропущено"
                )
                continue

            mu_y = float(y_clean[valid].mean())
            sd_y = float(y_clean[valid].std(ddof=1)) or 1.0
            mu_x = float(x_clean[valid].mean())
            sd_x = float(x_clean[valid].std(ddof=1)) or 1.0

            # Стандартизованные значения (только на valid)
            x_z = (x_clean[valid] - mu_x) / sd_x
            y_z = (y_clean[valid] - mu_y) / sd_y

            if abs(target) < 1e-9:
                # Простой разрыв: переставляем Y
                shuffled = y_z.to_numpy().copy()
                np_rng.shuffle(shuffled)
                y_new_z = shuffled
            else:
                # Y_new = target * X_z + sqrt(1 - target^2) * Y_shuffled_z
                # Корреляция X_z и Y_new_z равна target при независимом Y_shuffled_z.
                shuffled = y_z.to_numpy().copy()
                np_rng.shuffle(shuffled)
                coef = float(np.sqrt(max(0.0, 1.0 - target * target)))
                y_new_z = target * x_z.to_numpy() + coef * shuffled

            # Денормализация обратно к mean/std исходной col2
            y_new_vals = mu_y + sd_y * y_new_z
            corrupted_df.loc[valid[valid].index, p_cfg.col2] = y_new_vals.astype("float64")

            try:
                new_corr = float(corrupted_df[[p_cfg.col1, p_cfg.col2]].corr().iloc[0, 1])
            except Exception:
                new_corr = None

            applied_pairwise.append({
                "action": "break_correlation",
                "col1": p_cfg.col1,
                "col2": p_cfg.col2,
                "original_correlation": orig_corr,
                "new_correlation": new_corr,
                "target_correlation": p_cfg.target_correlation
            })
        elif p_cfg.action == "shift_dependency":
            # ТЕЗИС: "сдвиг зависимости" - изменить наклон линейной связи col2 = a*col1 + b + eps
            # до target_slope, сохранив маргинальное среднее/std col2 (форма зависимости меняется,
            # а маргинал col2 остаётся прежним по первым двум моментам).
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

            xv = x_clean[valid].to_numpy()
            yv = y_clean[valid].to_numpy()

            # ТЕЗИС: "сдвиг зависимости" - изменить наклон линейной связи при сохранении
            # маргинального mean/std col2. Делаем через стандартизованное смешивание:
            # slope = corr * std_y / std_x  =>  corr_target = target_slope * std_x / std_y.
            # Если |corr_target| > 1, цель недостижима при сохранении std_y.
            mu_x, sd_x = float(xv.mean()), float(xv.std(ddof=1)) or 1.0
            mu_y, sd_y = float(yv.mean()), float(yv.std(ddof=1)) or 1.0
            r_target = float(p_cfg.target_slope) * sd_x / sd_y

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

            # Исходный наклон (для отчёта)
            a_orig = float(np.cov(xv, yv, ddof=1)[0, 1] / (np.var(xv, ddof=1) or 1e-12))

            z_x = (xv - mu_x) / sd_x
            z_y = (yv - mu_y) / sd_y
            z_y_shuffled = z_y.copy()
            np_rng.shuffle(z_y_shuffled)
            coef = float(np.sqrt(max(0.0, 1.0 - r_target * r_target)))
            z_y_new = r_target * z_x + coef * z_y_shuffled

            # Денормализация к исходному mean/std col2
            y_new = mu_y + sd_y * z_y_new
            corrupted_df.loc[valid[valid].index, p_cfg.col2] = y_new.astype("float64")

            # Пересчитаем фактический наклон
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
            })
        elif p_cfg.action == "invert_correlation":
            # Инверсия знака (для числовых)
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

    # 3. Структурные искажения (ТЕЗИС: класс структурных - таблица в целом)
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
                    # Nullable Int64: переживает NaN, не роняя приведение
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
            # ТЕЗИС: "дрейф распределений" - сдвиг маргинала колонки целиком.
            # Y' = (Y + delta) * scale - форма распределения сохраняется,
            # но его положение и/или масштаб дрейфуют.
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
            # Приводим к float64 явно, чтобы избежать целочисленного переполнения
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
    exclude_columns = exclude_columns or []
    configs = {}

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
            # Раньше здесь был 'typo': подмена символов в дате даёт непарсируемые
            # строки. Пропуски контролируемы и сохраняют тип колонки.
            configs[col_name] = ColumnErrorConfig(error_type="missing", error_rate=error_rate, mechanism="ecar")

    return ErrorModel(column_configs=configs)

def save_error_model(model: ErrorModel, json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2, ensure_ascii=False)
