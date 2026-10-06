"""V02 去噪與銳化：人為加雜訊（知道答案），再用 PSNR 比較高斯／中值／雙邊濾波與 unsharp mask。

    python V02-denoise-sharpen/denoise.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import PHOTOS

OUT = Path("/tmp/ntunhs")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
clean = cv2.imread(str(PHOTOS / "bolts-conveyor.png"), cv2.IMREAD_GRAYSCALE)
THREAD = (slice(580, 700), slice(500, 600))  # 螺紋損傷那支的損傷處
BELT = (slice(400, 460), slice(650, 740))    # 一塊沒有東西的輸送帶


def psnr(a, b):
    return cv2.PSNR(a, b)


def gaussian_noise(g, sigma):
    return np.clip(g + rng.normal(0, sigma, g.shape), 0, 255).astype(np.uint8)


def salt_pepper(g, p):
    out = g.copy()
    r = rng.random(g.shape)
    out[r < p / 2] = 0
    out[r > 1 - p / 2] = 255
    return out


def detail(g):
    """細節量：Sobel 梯度強度的平均（邊緣與紋理越多越大；V05 會細講 Sobel）。"""
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1)
    return np.hypot(gx, gy).mean()


print("== 1. 先替 PSNR 建立直覺（越大越接近原圖，單位 dB）==")
for sigma in [5, 10, 20, 40]:
    print(f"高斯雜訊 σ={sigma:<2} → PSNR {psnr(clean, gaussian_noise(clean, sigma)):.2f} dB")

noisy = {"高斯 σ=20": gaussian_noise(clean, 20), "椒鹽 5%": salt_pepper(clean, 0.05)}
FILTERS = {
    "高斯 5×5": lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    "高斯 9×9": lambda x: cv2.GaussianBlur(x, (9, 9), 0),
    "中值 3": lambda x: cv2.medianBlur(x, 3),
    "中值 5": lambda x: cv2.medianBlur(x, 5),
    "雙邊 d=9": lambda x: cv2.bilateralFilter(x, 9, 50, 5),
}
print("\n== 2. 兩種雜訊 × 五種濾波：PSNR（dB）==")
rows = []
for nname, n in noisy.items():
    row = {"雜訊": nname, "不處理": psnr(clean, n)}
    for fname, f in FILTERS.items():
        row[fname] = psnr(clean, f(n))
    rows.append(row)
print(pd.DataFrame(rows).set_index("雜訊").round(2).to_string())

print("\n== 3. PSNR 之外：螺紋損傷處的細節還剩多少（原圖＝1.000）==")
base = detail(clean[THREAD])
n = noisy["高斯 σ=20"]
rows = [{"做法": "原圖", "損傷處細節": 1.0, "輸送帶雜訊": clean[BELT].std()},
        {"做法": "加雜訊", "損傷處細節": detail(n[THREAD]) / base, "輸送帶雜訊": n[BELT].std()}]
for fname, f in FILTERS.items():
    o = f(n)
    rows.append({"做法": fname, "損傷處細節": detail(o[THREAD]) / base, "輸送帶雜訊": o[BELT].std()})
print(pd.DataFrame(rows).set_index("做法").round(3).to_string())

print("\n== 4. unsharp mask：銳化 = 原圖 + amount × (原圖 − 模糊) ==")
soft = cv2.GaussianBlur(clean, (0, 0), 2)  # 假裝鏡頭失焦


def unsharp(g, amount, sigma=2):
    return cv2.addWeighted(g, 1 + amount, cv2.GaussianBlur(g, (0, 0), sigma), -amount, 0)


rows = []
for amount in [0, 0.5, 1, 2, 4, 8]:
    s = unsharp(soft, amount)
    rows.append({"amount": amount, "PSNR（對原圖）": psnr(clean, s), "細節（原圖＝1）": detail(s[THREAD]) / base,
                 "被截到 0 或 255": ((s == 0) | (s == 255)).mean()})
print(pd.DataFrame(rows).set_index("amount").round(3).to_string())
cv2.imwrite(str(OUT / "V02-unsharp-8.png"), unsharp(soft, 8))

print("\n== 5. 過度銳化的副作用：雜訊也被放大 ==")
soft_noisy = gaussian_noise(soft, 5)
rows = []
for name, im in [("失焦＋輕微雜訊", soft_noisy), ("直接銳化 amount=2", unsharp(soft_noisy, 2)),
                 ("先雙邊再銳化", unsharp(cv2.bilateralFilter(soft_noisy, 9, 30, 5), 2))]:
    rows.append({"做法": name, "PSNR": psnr(clean, im), "輸送帶雜訊": im[BELT].std(),
                 "損傷處細節": detail(im[THREAD]) / base})
print(pd.DataFrame(rows).set_index("做法").round(3).to_string())

print("\n== 6. 健康組：夜間走廊的低光雜訊 ==")
c = cv2.imread(str(PHOTOS / "care-corridor-fall.png"), cv2.IMREAD_GRAYSCALE)
night = np.clip(rng.poisson(c / 4.0) * 4.0, 0, 255).astype(np.uint8)  # 光子少：亮度越低、相對雜訊越大
print(f"模擬夜間雜訊後 PSNR {psnr(c, night):.2f} dB")
for fname in ["高斯 5×5", "中值 5", "雙邊 d=9"]:
    print(f"  {fname:<8} → {psnr(c, FILTERS[fname](night)):.2f} dB")
nl = cv2.fastNlMeansDenoising(night, None, h=12, templateWindowSize=7, searchWindowSize=21)
print(f"  {'NL-means':<8} → {psnr(c, nl):.2f} dB")

print("\n== 7. 挑錯：用 uint8 直接相減做銳化 ==")
blur = cv2.GaussianBlur(clean, (0, 0), 2)
wrong = clean + (clean - blur)  # uint8：負數會「繞回」變成 200 多
right = np.clip(clean.astype(np.int16) * 2 - blur, 0, 255).astype(np.uint8)
bad = np.abs(wrong.astype(int) - right.astype(int)) > 100
print(f"uint8 版與正確版差超過 100 的像素：{bad.mean():.3f}（{int(bad.sum())} 個）")
print(f"PSNR：uint8 版 {psnr(clean, wrong):.2f} dB、正確版 {psnr(clean, right):.2f} dB")
