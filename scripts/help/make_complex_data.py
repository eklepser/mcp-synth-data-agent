"""Generate a complex users CSV with mixed column semantics."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

n = 300

first_names = ["Ivan", "Petr", "Anna", "Maria", "Olga", "Sergey", "Dmitry", "Elena"]
last_names = ["Ivanov", "Petrov", "Smirnova", "Kuznetsov", "Popova", "Sokolov", "Lebedeva"]
cities = ["Moscow", "Saint Petersburg", "Kazan", "Novosibirsk", "Yekaterinburg", "Samara"]

ids = np.arange(1, n + 1)

full_names = [
    f"{rng.choice(first_names)} {rng.choice(last_names)}"
    for _ in range(n)
]

phones = [
    f"+7{rng.integers(900, 999)}{rng.integers(1000000, 9999999)}"
    for _ in range(n)
]

documents = rng.choice(np.arange(0, 1_000_000), size=n, replace=False)
documents = [f"{d:06d}" for d in documents]

emails = [
    f"user{i}@{rng.choice(['mail.ru', 'gmail.com', 'yandex.ru'])}"
    for i in ids
]

birth_dates = pd.to_datetime(
    rng.integers(
        pd.Timestamp("1960-01-01").value,
        pd.Timestamp("2005-12-31").value,
        size=n,
    )
).strftime("%Y-%m-%d")

signup_dates = pd.to_datetime(
    rng.integers(
        pd.Timestamp("2020-01-01").value,
        pd.Timestamp("2025-12-31").value,
        size=n,
    )
).strftime("%Y-%m-%d")

city = rng.choice(cities, size=n, p=[0.4, 0.25, 0.1, 0.1, 0.1, 0.05])

age = rng.integers(18, 90, size=n)

balance = np.round(rng.lognormal(mean=8.0, sigma=1.0, size=n), 2)

is_active = rng.choice([0, 1], size=n, p=[0.3, 0.7])

orders_count = rng.poisson(lam=3, size=n)

df = pd.DataFrame({
    "id": ids,
    "full_name": full_names,
    "phone": phones,
    "document": documents,
    "email": emails,
    "birth_date": birth_dates,
    "signup_date": signup_dates,
    "city": city,
    "age": age,
    "balance": balance,
    "is_active": is_active,
    "orders_count": orders_count,
})

out = RAW / "users_complex.csv"
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows and {df.shape[1]} columns")
print(df.head(3).to_string())