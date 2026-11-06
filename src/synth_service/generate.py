from __future__ import annotations

import random
import string
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

    if clip_to_range:
        for c in profile.columns:
            if c.kind == "numeric" and c.min is not None and c.max is not None:
                synth[c.name] = synth[c.name].clip(lower=c.min, upper=c.max)

    for c in profile.columns:
        if c.kind == "numeric" and c.is_integer and c.name in synth.columns:
            rounded = np.round(synth[c.name].to_numpy())
            if c.min is not None: rounded = np.maximum(rounded, np.floor(c.min))
            if c.max is not None: rounded = np.minimum(rounded, np.ceil(c.max))
            synth[c.name] = rounded.astype("int64")
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
    rng = np.random.default_rng(random_state)
    py_rng = random.Random(random_state)
    result = {}
    for c in profile.columns:
        if c.kind != "text":
            continue

        if c.text_pattern == "email":
            domains = c.text_options.get("domain", ["example.com"])
            emails = set()
            while len(emails) < n:
                num = rng.integers(1, 1000000)
                dom = py_rng.choice(domains)
                emails.add(f"user{num}@{dom}")
            result[c.name] = list(emails)

        elif c.text_pattern == "phone":
            length = c.text_options.get("length", 12)
            phones = set()
            while len(phones) < n:
                prefix = "+7" 
                rest_len = length - 2
                rest = "".join(str(py_rng.randint(0, 9)) for _ in range(rest_len))
                phones.add(f"{prefix}{rest}")
            result[c.name] = list(phones)

        elif c.text_pattern == "code":
            length = c.text_options.get("length", 6)
            codes = set()
            while len(codes) < n:
                val = py_rng.randint(0, 10**length - 1)
                codes.add(f"{val:0{length}d}")
            result[c.name] = list(codes)

        elif c.text_pattern == "full_name":
            f_names = c.text_options.get("first_name", ["Ivan"])
            l_names = c.text_options.get("last_name", ["Ivanov"])
            names = []
            for _ in range(n):
                names.append(f"{py_rng.choice(f_names)} {py_rng.choice(l_names)}")
            result[c.name] = names

        else:
            result[c.name] = ["".join(py_rng.choices(string.ascii_lowercase, k=8)) for _ in range(n)]

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