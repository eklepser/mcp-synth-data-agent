# src/synth_service/error_injector.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

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

def apply_numeric_typo(value: float, rng: random.Random) -> float:
    kind = rng.choice(["shift", "drop_zero", "duplicate_digit"])
    if kind == "shift":
        return value * rng.choice([0.1, 10.0])
    elif kind == "drop_zero":
        s = str(value)
        return float(s.replace("0", "", 1)) if "0" in s else value
    else:
        s = str(value)
        if len(s) < 2:
            return value
        idx = rng.randint(0, len(s) - 1)
        return float(s[:idx] + s[idx] + s[idx:])

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

# ============================================================================
# 2. МЕХАНИЗМЫ (ErrorMechanism)
# ============================================================================

def get_ecar_mask(n: int, error_rate: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n) < error_rate

def get_enar_mask(series: pd.Series, error_rate: float, condition: str, rng: np.random.Generator) -> np.ndarray:
    try:
        condition_mask = series.to_frame().eval(condition)
    except Exception as e:
        raise ValueError(f"Не удалось оценить условие '{condition}': {e}")
    base_mask = rng.random(len(series)) < error_rate
    return base_mask & condition_mask.values

# ============================================================================
# 3. МОДЕЛЬ ОШИБОК И ОТЧЕТ (ErrorModel & DistortionReport)
# ============================================================================

@dataclass
class ColumnErrorConfig:
    error_type: Literal["typo", "numeric_typo", "outlier", "missing", "category_swap"]
    error_rate: float
    mechanism: Literal["ecar", "enar"] = "ecar"
    condition: str | None = None

@dataclass
class PairwiseDistortionConfig:
    """Конфигурация для парных искажений (ТЕЗИС: разрыв корреляции, инверсия)"""
    col1: str
    col2: str
    action: Literal["break_correlation", "invert_correlation"]
    target_correlation: float = 0.0  # Для break_correlation цель ~0.0

@dataclass
class DistortionReport:
    """Структурированный отчет о примененных искажениях (ТЕЗИС: возвращая структурированный отчёт)"""
    target_error_rate: float
    actual_error_rate: float
    corrupted_cells_count: int
    applied_marginal_models: list[dict] = field(default_factory=list)
    applied_pairwise_actions: list[dict] = field(default_factory=list)
    structural_changes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ErrorModel:
    column_configs: dict[str, ColumnErrorConfig] = field(default_factory=dict)
    pairwise_configs: list[PairwiseDistortionConfig] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {"marginal": {}, "pairwise": []}
        for col, cfg in self.column_configs.items():
            result["marginal"][col] = asdict(cfg)
        for cfg in self.pairwise_configs:
            result["pairwise"].append(asdict(cfg))
        return result

def load_error_model(json_path: str) -> ErrorModel:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    configs = {}
    for col, cfg_dict in data.get("marginal", {}).items():
        configs[col] = ColumnErrorConfig(**cfg_dict)
    
    pairwise = [PairwiseDistortionConfig(**p) for p in data.get("pairwise", [])]
    return ErrorModel(column_configs=configs, pairwise_configs=pairwise)

# ============================================================================
# 4. ГЛАВНЫЙ ИНЖЕКТОР
# ============================================================================

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
    corrupted_count = 0
    applied_marginal = []

    # 1. Маргинальные искажения
    for col, config in model.column_configs.items():
        if col not in corrupted_df.columns:
            continue

        series = corrupted_df[col]
        non_null_mask = series.notna()

        if config.mechanism == "enar" and config.condition:
            mask = get_enar_mask(series, config.error_rate, config.condition, np_rng)
            mask = mask & non_null_mask.values
        else:
            mask = get_ecar_mask(len(series), config.error_rate, np_rng)
            mask = mask & non_null_mask.values

        error_mask.loc[mask, col] = True
        corrupted_count += int(mask.sum())
        applied_marginal.append({"column": col, "type": config.error_type, "rate": config.error_rate, "corrupted": int(mask.sum())})

        error_func = ERROR_TYPE_FUNCS.get(config.error_type)
        if error_func is None:
            raise ValueError(f"Неизвестный тип ошибки: {config.error_type}")

        for idx in series[mask].index:
            val = series.loc[idx]
            if config.error_type == "category_swap":
                valid_cats = series.dropna().unique().tolist()
                corrupted_df.loc[idx, col] = error_func(val, py_rng, valid_cats)
            else:
                corrupted_df.loc[idx, col] = error_func(val, py_rng)

    # 2. Парные и структурные искажения (ТЕЗИС: разрыв корреляции и др.)
    applied_pairwise = []
    structural_changes = {}

    for p_cfg in model.pairwise_configs:
        if p_cfg.col1 not in corrupted_df.columns or p_cfg.col2 not in corrupted_df.columns:
            continue
        
        # Вычисляем исходную корреляцию (только для числовых)
        try:
            orig_corr = float(corrupted_df[[p_cfg.col1, p_cfg.col2]].corr().iloc[0, 1])
        except Exception:
            orig_corr = None

        if p_cfg.action == "break_correlation":
            # Перемешиваем col2, сохраняя маргинальное распределение, но разрывая связь
            shuffled = corrupted_df[p_cfg.col2].sample(frac=1.0, random_state=np_rng).reset_index(drop=True)
            corrupted_df[p_cfg.col2] = shuffled
            
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
        elif p_cfg.action == "invert_correlation":
            # Инверсия знака (для числовых)
            if pd.api.types.is_numeric_dtype(corrupted_df[p_cfg.col2]):
                mean_val = corrupted_df[p_cfg.col2].mean()
                corrupted_df[p_cfg.col2] = mean_val - (corrupted_df[p_cfg.col2] - mean_val)
                applied_pairwise.append({"action": "invert_correlation", "col1": p_cfg.col1, "col2": p_cfg.col2})

    actual_error_rate = corrupted_count / total_cells if total_cells > 0 else 0.0

    report = DistortionReport(
        target_error_rate=sum(c.error_rate for c in model.column_configs.values()) / max(1, len(model.column_configs)),
        actual_error_rate=float(actual_error_rate),
        corrupted_cells_count=corrupted_count,
        applied_marginal_models=applied_marginal,
        applied_pairwise_actions=applied_pairwise,
        structural_changes=structural_changes
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
            configs[col_name] = ColumnErrorConfig(error_type="typo", error_rate=error_rate, mechanism="ecar")

    return ErrorModel(column_configs=configs)

def save_error_model(model: ErrorModel, json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2, ensure_ascii=False)