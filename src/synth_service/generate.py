from __future__ import annotations

import random
import string
import uuid
import numpy as np
import pandas as pd

from synth_service.copula_model import fit_copula
from synth_service.profile import TableProfile


def generate_numeric(df: pd.DataFrame, profile: TableProfile, n: int, random_state: int = 42, clip_to_range: bool = True) -> pd.DataFrame:
    numeric_cols = [c.name for c in profile.columns if c.kind == "numeric"]
    if not numeric_cols:
        return pd.DataFrame()

    copula = fit_copula(df, numeric_cols, random_state=random_state)
    synth = copula.sample(n)
    rng = np.random.default_rng(random_state)

    for c in profile.columns:
        if c.kind != "numeric":
            continue
            
        if c.is_integer:
            rounded = np.round(synth[c.name].to_numpy())
            if c.min is not None: 
                rounded = np.maximum(rounded, np.floor(c.min))
            if c.max is not None: 
                rounded = np.minimum(rounded, np.ceil(c.max))
            synth[c.name] = rounded.astype("int64")
        else:
            vals = synth[c.name].to_numpy()
            
            if c.min is not None and c.max is not None:
                cur_min, cur_max = np.min(vals), np.max(vals)
                if cur_max > cur_min:
                    vals = c.min + (vals - cur_min) * (c.max - c.min) / (cur_max - cur_min)
                else:
                    vals = np.full_like(vals, c.min)
                
                vals = np.clip(vals, c.min, c.max)

            if c.std is not None and c.std > 0:
                cur_std = np.std(vals)
                if cur_std > 0:
                    target_mean = c.mean if c.mean is not None else np.mean(vals)
                    vals = target_mean + (vals - np.mean(vals)) * (c.std / cur_std)
                    if c.min is not None and c.max is not None:
                        vals = np.clip(vals, c.min, c.max)

            span = (c.max - c.min) if (c.min is not None and c.max is not None) else 1.0
            delta = max(1e-5, span * 0.0001)
            vals = vals + rng.uniform(-delta, delta, size=len(vals))
            
            synth[c.name] = vals

    return synth


def generate_categorical(profile: TableProfile, n: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    result = {}
    for c in profile.columns:
        if c.kind == "categorical" and c.categories:
            cats = c.categories
            probs = np.array([c.category_freqs[str(k)] for k in cats])
            probs = probs / probs.sum()
            result[c.name] = rng.choice(cats, size=n, p=probs)
    return pd.DataFrame(result)


def generate_boolean(profile: TableProfile, n: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    result = {}
    for c in profile.columns:
        if c.kind == "boolean" and c.bool_values:
            vals = c.bool_values
            probs = np.array([c.bool_freqs[str(k)] for k in vals])
            probs = probs / probs.sum()
            result[c.name] = rng.choice(vals, size=n, p=probs)
    return pd.DataFrame(result)


def generate_identifier(df: pd.DataFrame, profile: TableProfile, n: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    result = {}
    for c in profile.columns:
        if c.kind == "identifier" and c.id_is_integer and c.id_min is not None:
            start = int(round(c.id_min))
            vals = np.arange(start, start + n, dtype=np.int64)
            rng.shuffle(vals)
            result[c.name] = vals
    return pd.DataFrame(result)


def generate_datetime(profile: TableProfile, n: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    result = {}
    for c in profile.columns:
        if c.kind == "datetime" and c.min is not None and c.max is not None:
            lo, hi = int(c.min), int(c.max)
            if hi <= lo:
                result[c.name] = np.full(n, lo, dtype="int64")
                continue
            offsets = rng.integers(lo, hi, size=n, dtype="int64")
            ts = pd.to_datetime(offsets, unit="ns")
            result[c.name] = ts.strftime(c.datetime_format) if c.datetime_format else ts
    return pd.DataFrame(result)


def generate_text(profile: TableProfile, n: int, random_state: int = 42) -> pd.DataFrame:
    py_rng = random.Random(random_state)
    result = {}
    for c in profile.columns:
        if c.kind != "text":
            continue

        if c.text_pattern == "uuid":
            generated = set()
            while len(generated) < n:
                generated.add(str(uuid.UUID(int=py_rng.getrandbits(128))))
            result[c.name] = list(generated)

        elif c.text_pattern == "mac":
            generated = set()
            while len(generated) < n:
                generated.add(":".join(f"{py_rng.randint(0, 255):02X}" for _ in range(6)))
            result[c.name] = list(generated)

        elif c.text_pattern == "phone":
            length = c.text_options.get("length", 12)
            generated = set()
            while len(generated) < n:
                prefix = "+7" 
                rest_len = length - 2
                rest = "".join(str(py_rng.randint(0, 9)) for _ in range(rest_len))
                generated.add(f"{prefix}{rest}")
            result[c.name] = list(generated)

        elif c.text_pattern == "pattern":
            pattern = c.text_options["pattern"]
            generated = set()
            max_attempts = n * 10 
            attempts = 0
            
            while len(generated) < n and attempts < max_attempts:
                attempts += 1
                s_parts = []
                for item in pattern:
                    item_type = item[0]
                    if item_type == "literal":
                        s_parts.append(item[1])
                    elif item_type == "number":
                        min_val, max_val, min_len, max_len = item[1], item[2], item[3], item[4]
                        val = py_rng.randint(min_val, max_val)
                        s_parts.append(f"{val:0{max_len}d}" if len(str(val)) < min_len else str(val))
                    elif item_type == "word":
                        words, is_lower, is_upper = item[1], item[2], item[3]
                        word = py_rng.choice(words)
                        if is_lower: word = word.lower()
                        elif is_upper: word = word.upper()
                        s_parts.append(word)
                    else:
                        min_len, max_len = item[1], item[2]
                        length = py_rng.randint(min_len, max_len)
                        s_parts.append("".join(py_rng.choice(string.ascii_letters) for _ in range(length)))
                generated.add("".join(s_parts))
            
            result[c.name] = list(generated)
            while len(result[c.name]) < n:
                result[c.name].append(py_rng.choice(result[c.name]))
                
        else:
            generated = set()
            while len(generated) < n:
                generated.add("".join(py_rng.choices(string.ascii_lowercase, k=8)))
            result[c.name] = list(generated)

    return pd.DataFrame(result)


def generate_table(df: pd.DataFrame, profile: TableProfile, n: int, random_state: int = 42) -> pd.DataFrame:
    parts = []
    parts.append(generate_numeric(df, profile, n, random_state))
    parts.append(generate_categorical(profile, n, random_state))
    parts.append(generate_boolean(profile, n, random_state))
    parts.append(generate_identifier(df, profile, n, random_state))
    parts.append(generate_datetime(profile, n, random_state))
    parts.append(generate_text(profile, n, random_state))

    synth = pd.concat([p for p in parts if not p.empty], axis=1)

    rng = np.random.default_rng(random_state + 1)
    for c in profile.columns:
        if c.name in synth.columns and c.missing_fraction > 0:
            mask = rng.random(n) < c.missing_fraction
            synth.loc[mask, c.name] = np.nan

    ordered = [c.name for c in profile.columns if c.name in synth.columns]
    return synth[ordered].reset_index(drop=True)