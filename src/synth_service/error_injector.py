"""
Модуль для внедрения реалистичных ошибок в синтетические данные.
Архитектура вдохновлена библиотекой tab_err.

Концепции:
- ErrorType: КАК искажается значение (typo, outlier, missing, category_swap)
- ErrorMechanism: ГДЕ появляются ошибки (ecar, ear, enar)
- ErrorModel: Конфигурация, объединяющая типы и механизмы для каждого столбца
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd


# ============================================================================
# 1. ТИПЫ ОШИБОК (ErrorType) - функции, которые искажают значение
# ============================================================================

def apply_typo(value: str, rng: random.Random) -> str:
    """Имитация опечатки: перестановка двух соседних символов."""
    if not isinstance(value, str) or len(value) < 2:
        return value
    idx = rng.randint(0, len(value) - 2)
    chars = list(value)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def apply_numeric_typo(value: float, rng: random.Random) -> float:
    """Сдвиг десятичной точки или потеря нуля: 1000 -> 100, 3.14 -> 31.4."""
    kind = rng.choice(["shift", "drop_zero", "duplicate_digit"])
    if kind == "shift":
        return value * rng.choice([0.1, 10.0])
    elif kind == "drop_zero":
        s = str(value)
        return float(s.replace("0", "", 1)) if "0" in s else value
    else:  # duplicate_digit
        s = str(value)
        if len(s) < 2:
            return value
        idx = rng.randint(0, len(s) - 1)
        return float(s[:idx] + s[idx] + s[idx:])


def apply_outlier(value: float, rng: random.Random) -> float:
    """Замена значения на статистический выброс."""
    multiplier = rng.choice([0.01, 0.1, 10.0, 100.0, -1.0])
    return value * multiplier


def apply_missing(_: Any, __: random.Random) -> float:
    """Замена значения на NaN."""
    return np.nan


def apply_category_swap(value: Any, rng: random.Random, valid_categories: list) -> Any:
    """Замена категории на другую из того же столбца."""
    alternatives = [v for v in valid_categories if v != value]
    if not alternatives:
        return value
    return rng.choice(alternatives)


# Словарь всех доступных типов ошибок
ERROR_TYPE_FUNCS = {
    "typo": apply_typo,
    "numeric_typo": apply_numeric_typo,
    "outlier": apply_outlier,
    "missing": apply_missing,
    "category_swap": apply_category_swap,
}


# ============================================================================
# 2. МЕХАНИЗМЫ (ErrorMechanism) - функции, определяющие ГДЕ ошибки
# ============================================================================

def get_ecar_mask(n: int, error_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Erroneous Completely At Random: полностью случайный выбор."""
    return rng.random(n) < error_rate


def get_enar_mask(series: pd.Series, error_rate: float, condition: str,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Erroneous Not At Random: ошибка зависит от значения.
    condition - строка вида "age < 18" или "salary > 100000"
    """
    try:
        # Используем pandas eval для безопасной оценки условия
        condition_mask = series.to_frame().eval(condition)
    except Exception as e:
        raise ValueError(f"Не удалось оценить условие '{condition}': {e}")
    
    base_mask = rng.random(len(series)) < error_rate
    return base_mask & condition_mask.values


# ============================================================================
# 3. МОДЕЛЬ ОШИБКИ (ErrorModel) - конфигурация
# ============================================================================

@dataclass
class ColumnErrorConfig:
    """Конфигурация ошибок для одного столбца."""
    error_type: Literal["typo", "numeric_typo", "outlier", "missing", "category_swap"]
    error_rate: float  # от 0.0 до 1.0
    mechanism: Literal["ecar", "enar"] = "ecar"
    condition: str | None = None  # Только для ENAR, например "age < 18"


@dataclass
class ErrorModel:
    """Модель ошибок для всей таблицы."""
    column_configs: dict[str, ColumnErrorConfig] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Сериализация в словарь для сохранения в JSON."""
        result = {}
        for col, cfg in self.column_configs.items():
            result[col] = {
                "error_type": cfg.error_type,
                "error_rate": cfg.error_rate,
                "mechanism": cfg.mechanism,
                "condition": cfg.condition,
            }
        return result


def load_error_model(json_path: str | Path) -> ErrorModel:
    """Загрузка модели ошибок из JSON-файла."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    configs = {}
    for col, cfg_dict in data.items():
        configs[col] = ColumnErrorConfig(
            error_type=cfg_dict["error_type"],
            error_rate=float(cfg_dict["error_rate"]),
            mechanism=cfg_dict.get("mechanism", "ecar"),
            condition=cfg_dict.get("condition"),
        )
    return ErrorModel(column_configs=configs)


# ============================================================================
# 4. ГЛАВНЫЙ ИНЖЕКТОР
# ============================================================================

def inject_errors(
    df: pd.DataFrame,
    model: ErrorModel,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Внедряет ошибки в DataFrame согласно модели.

    Returns:
        corrupted_df: DataFrame с искажёнными значениями
        error_mask: булев DataFrame (True = ячейка была изменена)
    """
    py_rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    corrupted_df = df.copy()
    error_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col, config in model.column_configs.items():
        if col not in corrupted_df.columns:
            continue

        series = corrupted_df[col]
        non_null_mask = series.notna()

        # 1. Определяем, ГДЕ будут ошибки (механизм)
        if config.mechanism == "enar" and config.condition:
            mask = get_enar_mask(series, config.error_rate, config.condition, np_rng)
            mask = mask & non_null_mask.values  # Не портим уже пропущенные
        else:
            mask = get_ecar_mask(len(series), config.error_rate, np_rng)
            mask = mask & non_null_mask.values

        error_mask.loc[mask, col] = True

        # 2. Применяем искажение (тип ошибки)
        error_func = ERROR_TYPE_FUNCS.get(config.error_type)
        if error_func is None:
            raise ValueError(f"Неизвестный тип ошибки: {config.error_type}")

        for idx in series[mask].index:
            val = series.loc[idx]
            if config.error_type == "category_swap":
                valid_cats = series.dropna().unique().tolist()
                corrupted_df.loc[idx, col] = error_func(val, py_rng, valid_cats)
            elif config.error_type in ("typo", "numeric_typo", "outlier", "missing"):
                corrupted_df.loc[idx, col] = error_func(val, py_rng)

    return corrupted_df, error_mask