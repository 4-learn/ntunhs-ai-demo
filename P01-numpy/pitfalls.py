"""P01｜讓 AI 寫，然後挑錯：AI 常寫的三個 NumPy 錯誤，實際跑一次。"""
import json

import numpy as np
import pandas as pd

from common import DATA

plc = pd.read_csv(DATA / "line/plc_a.csv")
vib, temp = plc["vib_rms"].to_numpy(), plc["temp_c"].to_numpy()
hr = np.array([json.loads(line)["hr"] for line in open(DATA / "care/wearable.jsonl", encoding="utf-8")])

print("== 錯誤 1：在整數陣列裡塞 NaN ==")
try:
    h = hr.copy()
    h[h == 0] = np.nan
except ValueError as e:
    print("ValueError:", e)
h = hr.astype(float)
h[h == 0] = np.nan
print(f"先 astype(float) 再改：NaN {int(np.isnan(h).sum())} 個")

print("\n== 錯誤 2：用 and 合併兩個條件 ==")
try:
    alarm = (vib > 1.5) and (temp > 50)
except ValueError as e:
    print("ValueError:", str(e).split(".")[0])
alarm = (vib > 1.5) & (temp > 50)
print(f"改用 &（記得括號）：同時偏高 {int(alarm.sum())} 秒")
try:
    no_paren = vib > 1.5 & (temp > 50)  # 優先順序：& 比 > 先算，變成 1.5 & 陣列
except TypeError:
    print("少了括號：TypeError（& 比 > 先算，變成 1.5 & 陣列）")

print("\n== 錯誤 3：遮罩後忘了 0 還在平均裡 ==")
print(f"hr.sum() / len(hr)       = {hr.sum() / len(hr):.3f}")
print(f"hr[hr > 0].mean()        = {hr[hr > 0].mean():.3f}")
print(f"np.nanmean(改成 NaN 的版) = {np.nanmean(h):.3f}")
