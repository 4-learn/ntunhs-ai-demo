"""V05 Sobel 與 Canny：門檻怎麼選，數邊緣像素比例；MVTec bottle 破口。

    python V05-edges/edges.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import PHOTOS, memory_calib_split, mvtec_paths

OUT = Path("/tmp/ntunhs")
OUT.mkdir(parents=True, exist_ok=True)

g = cv2.imread(str(PHOTOS / "bolts-conveyor.png"), cv2.IMREAD_GRAYSCALE)
print("== 1. Sobel：x 方向、y 方向的亮度變化 ==")
gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
mag = np.hypot(gx, gy)
wrong = cv2.Sobel(g, cv2.CV_8U, 1, 0, ksize=3)  # 負值被截成 0
print(f"gx 範圍 {gx.min():.0f} ～ {gx.max():.0f}；gy 範圍 {gy.min():.0f} ～ {gy.max():.0f}")
print(f"梯度強度：中位數 {np.median(mag):.1f}、99 百分位 {np.percentile(mag, 99):.1f}")
print(f"用 CV_8U 算 gx：保留到的邊緣 {(wrong > 50).sum()} 個像素；用 CV_64F 取絕對值：{(np.abs(gx) > 50).sum()} 個")
cv2.imwrite(str(OUT / "V05-sobel-mag.png"), np.clip(mag / 4, 0, 255).astype(np.uint8))

print("\n== 2. Canny 的兩個門檻：邊緣像素比例 ==")
H, W = g.shape
LEFT, RIGHT = (slice(0, H), slice(120, W // 3)), (slice(0, H), slice(2 * W // 3, W - 90))
rows = []
blur = cv2.GaussianBlur(g, (5, 5), 0)
for lo, hi in [(20, 60), (50, 150), (100, 200), (150, 300)]:
    for name, src in [("原圖", g), ("先高斯模糊", blur)]:
        e = cv2.Canny(src, lo, hi)
        rows.append({"low": lo, "high": hi, "輸入": name, "全圖邊緣比例": (e > 0).mean(),
                     "亮的左側": (e[LEFT] > 0).mean(), "暗的右側": (e[RIGHT] > 0).mean()})
print(pd.DataFrame(rows).set_index(["low", "high", "輸入"]).round(3).to_string())

print("\n== 3. 雙門檻的作用：同一個 high，low 決定邊緣能不能「接起來」==")
for lo in [30, 80, 140]:
    e = cv2.Canny(blur, lo, 150)
    n, _ = cv2.connectedComponents(e)
    print(f"Canny(low={lo:>3}, high=150)：邊緣像素 {(e > 0).sum():6d}、斷成 {n - 1:4d} 段")

print("\n== 4. MVTec bottle：破口的地方邊緣特別多嗎？==")
B = mvtec_paths("bottle")
S = 225


def load(p):
    return cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (S, S), interpolation=cv2.INTER_AREA)


def edges(p, lo=50, hi=150):
    return cv2.Canny(cv2.GaussianBlur(load(p), (5, 5), 0), lo, hi) > 0


rows = []
for d in ["broken_large", "broken_small", "contamination"]:
    inside, outside = [], []
    for p, t, m in zip(B["test"], B["defect"], B["mask"]):
        if t != d:
            continue
        mk = cv2.resize(cv2.imread(m, 0), (S, S), interpolation=cv2.INTER_NEAREST) > 127
        e = edges(p)
        inside.append(e[mk].mean())
        outside.append(e[~mk].mean())
    rows.append({"瑕疵": d, "張數": len(inside), "遮罩內邊緣比例": np.mean(inside), "遮罩外邊緣比例": np.mean(outside)})
print(pd.DataFrame(rows).set_index("瑕疵").round(3).to_string())

print("\n== 5. 「不該有邊緣的地方」出現邊緣：門檻換了，能抓到的瑕疵也變了 ==")
mem, cal = memory_calib_split(B["train"])
y = B["y"]
rows = []
for lo, hi in [(20, 60), (50, 150), (100, 200)]:
    # 記憶良品：每個位置「出現邊緣的頻率」；良品很少有邊緣的位置，出現邊緣才可疑
    freq = np.mean([cv2.dilate(edges(p, lo, hi).astype(np.uint8), np.ones((5, 5), np.uint8)) for p in mem], 0)
    score = lambda ps: np.array([(edges(p, lo, hi) & (freq < 0.05)).sum() for p in ps])  # noqa: E731
    s_cal, s_test = score(cal), score(B["test"])
    pred = s_test > s_cal.max()
    rows.append({"low": lo, "high": hi, "良品平均": s_test[y == 0].mean(), "瑕疵平均": s_test[y == 1].mean(),
                 "AUROC": roc_auc_score(y, s_test), "recall": pred[y == 1].mean(), "誤報": int(pred[y == 0].sum())})
print(pd.DataFrame(rows).set_index(["low", "high"]).round(3).to_string())
