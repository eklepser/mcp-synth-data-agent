import json
import sys
import math
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from synth_service.profile import profile_csv, _parse_str_column
from synth_service.generate import generate_table
from synth_service.error_injector import ErrorModel, error_model_from_dict, inject_errors
from synth_service.metrics import evaluate_reproduction, evaluate_distortion

mcp = FastMCP("SynthDataService")


def _clean_nan(obj):
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    elif isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64)):
        # Реальные выборки после парсинга содержат datetime64: sample_data
        # даёт pd.Timestamp, который json.dumps не сериализует.
        return None if pd.isna(obj) else str(obj)
    elif isinstance(obj, np.generic):
        # np.int64 / np.float64 / np.bool_ в JSON не сериализуются
        return obj.item()
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _safe_json_dumps(data, **kwargs) -> str:
    return json.dumps(_clean_nan(data), **kwargs, allow_nan=False)


def _error_model_from_rules_json(rules_json: str) -> tuple[ErrorModel | None, str | None]:
    """Разбирает rules_json в ErrorModel (общий путь для всех тулов искажений).

    Поддерживает полный формат {"marginal": {...}, "pairwise": [...], "structural": [...]}
    и плоский {колонка: {...}} (нормализуется в {"marginal": ...}, ключ rate -> error_rate).
    Возвращает (model, None) либо (None, текст ошибки) - тулы переводят его в {"error": ...}.
    """
    try:
        rules = json.loads(rules_json)
    except ValueError as e:
        return None, f"rules_json не является корректным JSON: {e}"
    if not isinstance(rules, dict):
        return None, "rules_json должен быть JSON-объектом"

    # Совместимость со старым форматом тула: плоский {колонка: {..., "rate": ...}}
    # нормализуем в {"marginal": ...} и далее единый путь.
    if "marginal" not in rules and "pairwise" not in rules and "structural" not in rules:
        flat = {}
        for col, rule in rules.items():
            if not isinstance(rule, dict):
                return None, f"Правило для колонки '{col}' должно быть JSON-объектом"
            flat[col] = {("error_rate" if k == "rate" else k): v for k, v in rule.items()}
        rules = {"marginal": flat}

    try:
        return error_model_from_dict(rules), None
    except (ValueError, TypeError) as e:
        return None, f"Некорректная модель искажений: {e}"


def _ensure_output_dirs() -> None:
    """data/generated отсутствует в свежем клоне (в .gitignore) - создаём на месте."""
    Path("data/generated").mkdir(parents=True, exist_ok=True)


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
    """Профилирует CSV и возвращает JSON-строку с типами колонок.

    csv_path - имя файла из data/raw/ (например, из list_available_files)
    или путь, начинающийся с "data/".
    """
    try:
        # Нормализация пути - та же, что в остальных тулах: агент передаёт
        # имя файла из list_available_files без префикса data/raw/
        if not str(csv_path).startswith("data/"):
            csv_path = f"data/raw/{csv_path}"
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

        _ensure_output_dirs()
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
         {"col1", "col2", "action": "break_correlation" | "false_link" | "invert_correlation" | "shift_dependency"}
         Для "break_correlation" можно задать "target_correlation" в [-1, 1]:
           - 0.0 (по умолчанию) - полный разрыв через перестановку col2;
           - любое значение в (-1, 1) - ослабление/усиление до целевой корреляции,
             сохраняя маргинальное mean/std col2.
         Для "false_link" можно задать "target_correlation" в [-1, 1] (по умолчанию 0.5):
           - создает ложную связь заданной силы между col1 и col2 там, где её
             не было (класс "ложная связь"), сохраняя маргинальное mean/std col2.
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
         {"action": "violate_constraints", "columns": ["id"], "constraint": "unique", "rate": 0.1}
           - нарушение ограничений (ТЕЗИС): "unique" - нарушает уникальность колонки
             (создаёт дубликаты значений); "range" - вставляет значения вне заявленного
             допустимого диапазона (обязательно min_value и/или max_value; ровно одна
             колонка в columns); "not_null" - вставляет NaN в обязательную колонку
             (int -> float64, bool -> object).

    2) ПЛОСКИЙ (совместимость): {колонка: {"error_type", "rate", "mechanism", "condition",
       "intensity"}} - то же, что "marginal" без обёртки, ключ "rate" равнозначен "error_rate".

    В ответе: distortion_report (план/факт по каждой модели, предупреждения, пропуски,
    структурные изменения), applied_model - итоговая применённая модель искажений,
    distortion_evaluation - метрики режима искажения (прирост дисперсии, сдвиг
    корреляционной матрицы, выполнение целей парных искажений и совпадение схемы
    с целевой: schema_matches_target / schema_match_score / schema_checks).
    """
    try:
        if not str(source_path).startswith("data/"):
            source_path = f"data/raw/{source_path}"

        model, model_error = _error_model_from_rules_json(rules_json)
        if model is None:
            return _safe_json_dumps({"error": model_error})

        df_str = pd.read_csv(source_path, dtype=str)
        df = df_str.copy()
        for col in df_str.columns:
            df[col] = _parse_str_column(df_str[col])

        prof = profile_csv(source_path)
        synth_clean = generate_table(df, prof, n=rows, random_state=random_state)

        synth_noisy, mask, report = inject_errors(synth_clean, model, random_state=random_state)

        _ensure_output_dirs()
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


@mcp.tool()
def distort_existing_csv(csv_path: str, rules_json: str, random_state: int = 42) -> str:
    """Применяет контролируемые искажения к СУЩЕСТВУЮЩЕЙ выборке (реальной или синтетической), без генерации.

    Используйте этот тул, когда нужно испортить уже готовый CSV-файл:
    реальную выборку или ранее сгенерированный файл. Если данные нужно
    сначала сгенерировать, а затем исказить - используйте generate_custom_distortions.

    csv_path - имя файла из data/raw/ или путь, начинающийся с "data/".
    rules_json - JSON со спецификацией искажений; формат полностью совпадает
    с rules_json тула generate_custom_distortions:
      {"marginal": {колонка: {"error_type", "error_rate", "mechanism", "condition", "intensity"}},
       "pairwise": [{"col1", "col2", "action", "target_correlation"?, "target_slope"?}],
       "structural": [{"action", "columns"?, "rate"?, "constraint"?, "min_value"?, "max_value"?}]},
    допустим и плоский формат {колонка: {"error_type", "rate", ...}}.

    В ответе: corrupted_csv - путь к испорченному файлу, error_mask_csv - карта
    искажённых ячеек (1 - ячейка изменена), distortion_report (план/факт по каждой
    модели, warnings, skipped, структурные изменения), distortion_evaluation -
    метрики режима искажения (прирост дисперсии, сдвиг корреляционной матрицы,
    выполнение целей парных искажений, совпадение схемы с целевой:
    schema_matches_target / schema_match_score / schema_checks) и applied_model.
    """
    try:
        if not str(csv_path).startswith("data/"):
            csv_path = f"data/raw/{csv_path}"
        if not Path(csv_path).exists():
            return _safe_json_dumps({"error": f"Файл не найден: {csv_path}"})

        model, model_error = _error_model_from_rules_json(rules_json)
        if model is None:
            return _safe_json_dumps({"error": model_error})

        df_str = pd.read_csv(csv_path, dtype=str)
        df = df_str.copy()
        for col in df_str.columns:
            df[col] = _parse_str_column(df_str[col])

        # ТЕЗИС: "В режиме искажения источником может служить как реальная,
        # так и синтетическая выборка" - здесь источник подаётся на вход
        # инжектору напрямую, без этапа генерации.
        corrupted_df, error_mask, report = inject_errors(df, model, random_state=random_state)

        _ensure_output_dirs()
        stem = Path(csv_path).stem
        out_path = f"data/generated/{stem}_corrupted.csv"
        mask_path = f"data/generated/{stem}_corrupted_mask.csv"
        corrupted_df.to_csv(out_path, index=False)
        error_mask.astype(int).to_csv(mask_path, index=False)

        # ТЕЗИС: "В режиме искажения измеряется соответствие фактического отклонения
        # заданному: прирост дисперсии, сдвиг корреляционной матрицы, совпадение
        # схемы с целевой." Оценка считается относительно исходной (реальной) выборки.
        distortion_eval = evaluate_distortion(df, corrupted_df, report)

        sample_df = corrupted_df.head(50)
        if "id" in sample_df.columns:
            try:
                sample_df = sample_df.sort_values("id")
            except Exception:
                pass
        sample_data = _clean_nan(sample_df.head(10).to_dict(orient="records"))

        result = {
            "source_file": str(csv_path),
            "corrupted_csv": str(out_path),
            "error_mask_csv": str(mask_path),
            "source_rows": len(df),
            "corrupted_rows": len(corrupted_df),
            "mode": "distort_existing",
            "distortion_report": report.to_dict(),
            "distortion_evaluation": distortion_eval.to_dict(),
            "applied_model": model.to_dict(),
            "sample_data": sample_data,
        }
        return _safe_json_dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return _safe_json_dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()