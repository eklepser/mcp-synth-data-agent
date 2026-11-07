"""Generate the ultimate complex dataset covering all edge cases."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

n = 500

# 1. Идентификатор (плотный integer identifier)
user_id = np.arange(1, n + 1)

# 2. Обычные числа (вещественные)
age = rng.integers(18, 90, size=n).astype(float)  # на самом деле целые, но проверим is_integer
balance = np.round(rng.lognormal(mean=8.0, sigma=1.0, size=n), 2)

# 3. Строгий UUID (8-4-4-4-12, только hex)
def make_uuid():
    return f"{rng.integers(0, 16**8):08x}-{rng.integers(0, 16**4):04x}-{rng.integers(0, 16**4):04x}-{rng.integers(0, 16**4):04x}-{rng.integers(0, 16**12):012x}"
session_id = [make_uuid() for _ in range(n)]

# 4. IP-адреса (числа строго от 0 до 255)
ip_address = [f"{rng.integers(1, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for _ in range(n)]

# 5. Артикулы (SKU) со смешанным регистром
skus = [f"SKU-{rng.integers(1000, 9999)}-{rng.choice(['A', 'B', 'C'])}" for _ in range(n)]

# 6. MAC-адреса (фиксированная длина, hex в верхнем регистре, двоеточия)
macs = [":".join(f"{rng.integers(0, 255):02X}" for _ in range(6)) for _ in range(n)]

# 7. Сложные адреса (смесь слов, запятых, точек и чисел)
cities = ["Moscow", "Saint Petersburg", "Kazan", "Novosibirsk", "Yekaterinburg"]
streets = ["Tverskaya st.", "Lenina st.", "Baumana st.", "Gagarina st.", "Mira st."]
addresses = [f"{rng.choice(cities)}, {rng.choice(streets)}, {rng.integers(1, 100)}" for _ in range(n)]

# 8. Дата и время (время с часами, минутами и секундами)
datetimes = pd.to_datetime(
    rng.integers(pd.Timestamp("2020-01-01").value, pd.Timestamp("2024-12-31").value, size=n)
).strftime("%Y-%m-%d %H:%M:%S")

# 9. Число с ведущими нулями (документ/код)
codes = [f"{rng.integers(0, 999999):06d}" for _ in range(n)]

# 10. Телефон с плюсом
phones = [f"+7{rng.integers(900, 999)}{rng.integers(1000000, 9999999)}" for _ in range(n)]

# 11. Email (с разными доменами)
emails = [f"user{rng.integers(1, 10000)}@{rng.choice(['yandex.ru', 'mail.ru', 'gmail.com'])}" for _ in range(n)]

# 12. Категориальные данные
status = rng.choice(["active", "inactive", "banned"], size=n, p=[0.7, 0.2, 0.1])

# 13. Логические данные
is_verified = rng.choice([0, 1], size=n, p=[0.3, 0.7])

# 14. Текст с пунктуацией (названия компаний)
companies = [f"{rng.choice(['LLC', 'Inc', 'Corp'])} '{rng.choice(['Alpha', 'Beta', 'Gamma'])}'" for _ in range(n)]

# 15. URL с параметрами (смесь символов, цифр и знаков)
urls = [f"https://{rng.choice(['shop.com', 'store.ru'])}/item/{rng.integers(100, 999)}?ref={rng.choice(['google', 'yandex'])}" for _ in range(n)]

df = pd.DataFrame({
    "id": user_id,
    "session_uuid": session_id,
    "ip_address": ip_address,
    "mac_address": macs,
    "sku": skus,
    "address": addresses,
    "signup_time": datetimes,
    "document_code": codes,
    "phone": phones,
    "email": emails,
    "company_name": companies,
    "url": urls,
    "age": age,
    "balance": balance,
    "status": status,
    "is_verified": is_verified,
})

out = RAW / "ultimate_complex.csv"
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows and {df.shape[1]} columns")
print(df.head(3).to_string())