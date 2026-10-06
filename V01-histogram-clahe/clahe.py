"""V01 直方圖與 CLAHE：用數字比較「原圖、全域等化、CLAHE」在打光不均照片上的效果。

    python V01-histogram-clahe/clahe.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import PHOTOS

OUT = Path("/tmp/ntunhs")
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["左上", "中上", "右上", "左中", "正中", "右中", "左下", "中下", "右下"]
BELT = (slice(400, 460), slice(650, 740))  # 輸送帶上沒有螺栓的一塊（看雜訊用）


def regions(g):
    """切成 3×3 九宮格，回傳每一格的亮度平均與標準差（對比）。"""
    h, w = g.shape
    cells = [g[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3] for i in range(3) for j in range(3)]
    return np.array([c.mean() for c in cells]), np.array([c.std() for c in cells])


def report(g, belt=None):
    m, s = regions(g)
    row = {"整體平均": g.mean(), "分區平均最亮": m.max(), "分區平均最暗": m.min(),
           "亮暗差": m.max() - m.min(), "最暗區對比": s[m.argmin()], "全黑或全白比例": ((g == 0) | (g == 255)).mean()}
    if belt is not None:
        row["輸送帶雜訊"] = g[belt].std()
    return row


def clahe(g, clip=2.0, tile=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)


g = cv2.imread(str(PHOTOS / "bolts-conveyor.png"), cv2.IMREAD_GRAYSCALE)
print("== 1. 原圖：九宮格亮度平均 ==")
m, s = regions(g)
print(pd.DataFrame(m.reshape(3, 3).round(1), index=["上", "中", "下"], columns=["左", "中", "右"]).to_string())
p = np.percentile(g, [1, 50, 99])
print(f"亮度百分位 1%／50%／99%：{p[0]:.0f}／{p[1]:.0f}／{p[2]:.0f}（0–255 只用了一小段）")

eq = cv2.equalizeHist(g)
cl = clahe(g)
print("\n== 2. 全域等化 vs CLAHE（clipLimit=2, 8×8）==")
tab = pd.DataFrame({"原圖": report(g, BELT), "全域等化": report(eq, BELT), "CLAHE": report(cl, BELT)}).T
print(tab.round(3).to_string())
for name, im in [("orig", g), ("eq", eq), ("clahe", cl)]:
    cv2.imwrite(str(OUT / f"V01-bolts-{name}.png"), im)

print("\n== 3. 局部對比：螺栓亮度 − 旁邊輸送帶亮度 ==")
ROI = {"亮區螺栓": (250, 270), "亮區輸送帶": (150, 380), "暗區螺栓": (1200, 600), "暗區輸送帶": (1320, 420)}
rows = []
for name, im in [("原圖", g), ("全域等化", eq), ("CLAHE", cl)]:
    v = {k: im[y - 25:y + 25, x - 25:x + 25].mean() for k, (x, y) in ROI.items()}
    rows.append({"做法": name, "亮區對比": v["亮區螺栓"] - v["亮區輸送帶"], "暗區對比": v["暗區螺栓"] - v["暗區輸送帶"]})
print(pd.DataFrame(rows).set_index("做法").round(1).to_string())

print("\n== 4. clipLimit 越大，暗處越清楚，雜訊也越大 ==")
rows = []
for clip in [1, 2, 4, 8]:
    r = report(clahe(g, clip), BELT)
    rows.append({"clipLimit": clip, "最暗區對比": r["最暗區對比"], "亮暗差": r["亮暗差"], "輸送帶雜訊": r["輸送帶雜訊"]})
print(pd.DataFrame(rows).set_index("clipLimit").round(3).to_string())

print("\n== 5. 彩色照片：只對亮度做 CLAHE ==")
c = cv2.imread(str(PHOTOS / "care-corridor-fall.png"))
lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)
good = cv2.cvtColor(cv2.merge([clahe(L), A, B]), cv2.COLOR_LAB2BGR)
bad = cv2.merge([clahe(ch) for ch in cv2.split(c)])
for name, im in [("原圖", c), ("Lab 只處理 L", good), ("B、G、R 各做一次", bad)]:
    lab2 = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(float)
    shift = np.abs(lab2[..., 1:] - lab[..., 1:].astype(float)).mean()
    print(f"{name:<12} 亮度平均 {lab2[..., 0].mean():6.1f}  色偏（a、b 平均變化）{shift:.3f}")
cv2.imwrite(str(OUT / "V01-corridor-clahe.png"), good)

print("\n== 6. 夜間走廊（健康組）：同一把尺 ==")
gc = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
tab = pd.DataFrame({"原圖": report(gc), "全域等化": report(cv2.equalizeHist(gc)), "CLAHE": report(clahe(gc))}).T
print(tab.round(3).to_string())
