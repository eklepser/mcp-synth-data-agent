"""Generate the three baseline CSV datasets in data/raw/."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

n = 500
x = rng.normal(0, 1, n)
y = 0.7 * x + rng.normal(0, 0.5, n)
z = -0.4 * x + rng.normal(0, 0.8, n)
pd.DataFrame({"x": x, "y": y, "z": z}).to_csv(RAW / "numeric_correlated.csv", index=False)

n = 400
cat = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
num1 = rng.normal(10, 2, n)
num2 = rng.normal(5, 1, n)
num2_missing = num2.copy()
mask = rng.random(n) < 0.12
num2_missing[mask] = np.nan
pd.DataFrame({"category": cat, "num1": num1, "num2": num2_missing}).to_csv(
    RAW / "mixed_types.csv", index=False
)

n = 300
price = rng.lognormal(mean=3.0, sigma=0.5, size=n)
qty = rng.integers(1, 20, size=n)
total = price * qty
pd.DataFrame({"price": price, "qty": qty, "total": total}).to_csv(
    RAW / "constrained.csv", index=False
)

print("Created datasets:")
for p in sorted(RAW.glob("*.csv")):
    print(f"  {p.name}: {sum(1 for _ in p.open()) - 1} rows")