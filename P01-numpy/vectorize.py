"""P01｜NumPy：一次算一整欄——向量化、廣播、布林遮罩（產線振動、手環心率）。"""
import json
import time

import numpy as np
import pandas as pd

from common import DATA

plc = pd.read_csv(DATA / "line/plc_a.csv")
vib = plc["vib_rms"].to_numpy()
temp = plc["temp_c"].to_numpy()

print("== 1. 一欄就是一個陣列 ==")
print(f"PLC A：{len(plc)} 列；vib 的型別 {type(vib).__name__}、shape {vib.shape}、dtype {vib.dtype}")
print("前 5 筆振動：", vib[:5].tolist())
print(f"平均 {vib.mean():.3f}、最大 {vib.max():.3f}、標準差 {vib.std():.3f}")

print("\n== 2. for 迴圈 vs 向量化：振動比基準高了幾 % ==")
base = float(np.median(vib[:3600]))          # 第一小時（刀具還新）的中位數當基準
print(f"基準（第一小時中位數）{base:.3f}")


def ratio_loop(v, b):
    out = []
    for x in v:
        out.append((x - b) / b * 100)
    return out


t0 = time.perf_counter()
r_loop = np.array(ratio_loop(vib, base))
t_loop = time.perf_counter() - t0
t0 = time.perf_counter()
r_vec = (vib - base) / base * 100
t_vec = time.perf_counter() - t0
print("兩種寫法結果一致：", np.allclose(r_loop, r_vec))
print("前 3 筆：", np.round(r_vec[:3], 3).tolist(), " 最大：", round(float(r_vec.max()), 3), "%")
# 速度會因機器而異，講義不貼這一行
print(f"（速度參考）for 迴圈約是向量化的 {t_loop / max(t_vec, 1e-9):.0f} 倍時間")

print("\n== 3. 廣播：兩欄一起標準化 ==")
X = np.column_stack([temp, vib])
mu, sd = X.mean(axis=0), X.std(axis=0)
Z = (X - mu) / sd
print(f"X {X.shape} 減去 mu {mu.shape} → {Z.shape}")
print("每欄平均：", np.round(Z.mean(axis=0), 3).tolist(), " 每欄標準差：", np.round(Z.std(axis=0), 3).tolist())
print("°C → °F 也是廣播（陣列 × 純量）：", np.round(temp[:3] * 9 / 5 + 32, 2).tolist())

print("\n== 4. 布林遮罩：把「振動偏高」的秒挑出來 ==")
high = vib > 1.5
print(f"high 的 dtype {high.dtype}；True 有 {high.sum()} 秒，占 {high.mean():.3f}")
hours = pd.to_datetime(plc["time"]).dt.hour.to_numpy()
for h in np.unique(hours):
    m = hours == h
    print(f"  {h:02d} 點：平均振動 {vib[m].mean():.3f}，偏高 {int((high & m).sum()):4d} 秒")

print("\n== 5. 手環心率：hr=0 不是心跳停了，是沒量到 ==")
recs = [json.loads(line) for line in open(DATA / "care/wearable.jsonl", encoding="utf-8")]
hr = np.array([r["hr"] for r in recs])
dev = np.array([r["device"] for r in recs])
print(f"手環 {len(hr)} 筆；hr dtype {hr.dtype}；hr=0 有 {(hr == 0).sum()} 筆")
valid = hr > 0
print(f"直接平均（含 0）  {hr.mean():.3f}")
print(f"遮罩後平均（hr>0）{hr[valid].mean():.3f}")
hr_f = np.where(valid, hr, np.nan)           # 先變 float，0 改成 NaN
print(f"改成 NaN 再 nanmean {np.nanmean(hr_f):.3f}；dtype {hr_f.dtype}")
m = dev == "WB-011"
print(f"WB-011 最低心率：含 0 → {hr[m].min()}；遮罩後 → {hr[m & valid].min()}")
