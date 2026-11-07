"""Generate a second complex dataset with SKUs, IPs, addresses, and URLs."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

n = 400

# 1. Артикулы (SKU): смесь букв и цифр, фиксированный формат
skus = [f"SKU-{rng.integers(1000, 9999)}-{rng.choice(['A', 'B', 'C'])}" for _ in range(n)]

# 2. IP-адреса: только цифры и точки
ips = [f"{rng.integers(1, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for _ in range(n)]

# 3. Адреса: смесь текста, цифр, запятых и точек
cities = ["Moscow", "Saint Petersburg", "Kazan", "Novosibirsk", "Yekaterinburg"]
streets = ["Tverskaya st.", "Lenina st.", "Baumana st.", "Gagarina st.", "Mira st."]
addresses = [f"{rng.choice(cities)}, {rng.choice(streets)}, {rng.integers(1, 100)}" for _ in range(n)]

# 4. URL-адреса: протокол, домен, путь
domains = ["shop.com", "store.ru", "market.net", "example.org"]
paths = ["/product/123", "/category/electronics", "/cart", "/checkout", "/user/profile"]
urls = [f"https://{rng.choice(domains)}{rng.choice(paths)}" for _ in range(n)]

# 5. UUID: строгий hex-формат 8-4-4-4-12
uuids = [f"{rng.integers(0, 16**8):08x}-{rng.integers(0, 16**4):04x}-{rng.integers(0, 16**4):04x}-{rng.integers(0, 16**4):04x}-{rng.integers(0, 16**12):012x}" for _ in range(n)]

# 6. Дата регистрации и количество заказов (обычные данные для сравнения)
signup_dates = pd.to_datetime(
    rng.integers(pd.Timestamp("2020-01-01").value, pd.Timestamp("2024-12-31").value, size=n)
).strftime("%Y-%m-%d")

orders = rng.integers(0, 50, size=n)

# 7. Статус (категориальный)
status = rng.choice(["active", "inactive", "banned"], size=n, p=[0.7, 0.2, 0.1])

df = pd.DataFrame({
    "user_id": np.arange(1000, 1000 + n),
    "sku": skus,
    "ip_address": ips,
    "address": addresses,
    "url": urls,
    "session_id": uuids,
    "signup_date": signup_dates,
    "orders_count": orders,
    "status": status,
})

out = RAW / "ecommerce_complex.csv"
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows and {df.shape[1]} columns")
print(df.head(5).to_string())