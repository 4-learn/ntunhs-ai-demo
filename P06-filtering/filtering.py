"""P06｜濾波：手算 3×3 卷積 = cv2.filter2D；平均、高斯、中值、Sobel；雜訊與模糊的取捨。"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import PHOTOS

OUT = Path("/tmp/ntunhs/P06")
OUT.mkdir(parents=True, exist_ok=True)
np.set_printoptions(linewidth=120)

bolts = cv2.imread(str(PHOTOS / "bolts-conveyor.png"))
gray = cv2.cvtColor(bolts, cv2.COLOR_BGR2GRAY)
crop = gray[470:710, 460:610]                     # 第 2 列中間：螺紋損傷的螺栓（P02 的框）

print("== 1. 手算一個位置：3×3 窗口 × 核，逐格相乘再加總 ==")
y, x = 120, 75
win = crop[y - 1:y + 2, x - 1:x + 2].astype(np.float32)
mean_k = np.ones((3, 3), np.float32) / 9
print("窗口（螺栓上的 3×3 像素）：")
print(win.astype(int))
print("平均核：每格 1/9")
hand = float((win * mean_k).sum())
cv_val = cv2.filter2D(crop.astype(np.float32), -1, mean_k)[y, x]
print(f"手算 {hand:.3f}；cv2.filter2D 同一位置 {cv_val:.3f}")

print("\n== 2. 手寫整張卷積（兩層 for 迴圈）vs cv2.filter2D ==")
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], np.float32)


def conv2d_loop(img, k):
    img = img.astype(np.float32)
    h, w = img.shape
    out = np.zeros((h - 2, w - 2), np.float32)     # 不補邊：輸出少一圈
    for i in range(h - 2):
        for j in range(w - 2):
            out[i, j] = (img[i:i + 3, j:j + 3] * k).sum()
    return out


small = crop[100:140, 50:100]
mine = conv2d_loop(small, sobel_x)
cv = cv2.filter2D(small.astype(np.float32), cv2.CV_32F, sobel_x)[1:-1, 1:-1]
print(f"輸入 {small.shape} → 手寫輸出 {mine.shape}（少一圈）；cv2 去掉外圈後 {cv.shape}")
print(f"兩者最大差 {np.abs(mine - cv).max():.6f}")
print(f"手寫值域 {mine.min():.0f} ~ {mine.max():.0f}（有負數）")

print("\n== 3. cv2.filter2D 不翻轉核：它其實是「相關」==")
flipped = cv2.filter2D(small.astype(np.float32), cv2.CV_32F, sobel_x[::-1, ::-1].copy())[1:-1, 1:-1]
print(f"把核上下左右翻轉再算：和原本的相關係數 {np.corrcoef(cv.ravel(), flipped.ravel())[0, 1]:.3f}")
print("對稱的核（平均、高斯）翻不翻都一樣；Sobel 翻轉後正負號整個相反")

print("\n== 4. 四種核長什麼樣 ==")
g1 = cv2.getGaussianKernel(3, 0)
print("平均 3×3：每格", f"{1 / 9:.3f}")
print("高斯 3×3：\n", np.round(g1 @ g1.T, 3))
print("Sobel x（找左右亮度變化 = 直的邊）：\n", sobel_x.astype(int))
print("Sobel y（找上下亮度變化 = 橫的邊）：\n", sobel_x.T.astype(int))
print("中值濾波沒有核：取窗口內 9 個數的中位數（不是乘加，不是卷積）")

print("\n== 5. 加雜訊，再用平均、高斯、中值去掉 ==")
rng = np.random.default_rng(0)
clean = gray[400:1000, 300:1100]
gauss_noisy = np.clip(clean + rng.normal(0, 20, clean.shape), 0, 255).astype(np.uint8)
sp_noisy = clean.copy()
m = rng.random(clean.shape)
sp_noisy[m < 0.03] = 0
sp_noisy[m > 0.97] = 255


def edge_strength(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


filters = {
    "不處理": lambda im: im,
    "平均 5×5": lambda im: cv2.blur(im, (5, 5)),
    "高斯 5×5": lambda im: cv2.GaussianBlur(im, (5, 5), 0),
    "中值 3×3": lambda im: cv2.medianBlur(im, 3),
    "中值 5×5": lambda im: cv2.medianBlur(im, 5),
}
rows = []
for name, f in filters.items():
    rows.append({"濾波": name,
                 "高斯雜訊_誤差": round(float(cv2.absdiff(f(gauss_noisy), clean).mean()), 3),
                 "椒鹽雜訊_誤差": round(float(cv2.absdiff(f(sp_noisy), clean).mean()), 3),
                 "乾淨圖_邊緣強度": round(edge_strength(f(clean)), 3)})
print(pd.DataFrame(rows).to_string(index=False))
cv2.imwrite(str(OUT / "sp-noisy.png"), sp_noisy)
cv2.imwrite(str(OUT / "sp-median.png"), cv2.medianBlur(sp_noisy, 5))
cv2.imwrite(str(OUT / "sp-mean.png"), cv2.blur(sp_noisy, (5, 5)))

print("\n== 6. 核越大越平滑，邊也越糊（高斯，高斯雜訊）==")
for k in [3, 5, 9, 15, 25]:
    out = cv2.GaussianBlur(gauss_noisy, (k, k), 0)
    print(f"{k:>2}×{k:<2}  與乾淨圖誤差 {cv2.absdiff(out, clean).mean():6.3f}   邊緣強度 {edge_strength(out):6.3f}")

print("\n== 7. Sobel：邊的方向 ==")
gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
mag = np.sqrt(gx ** 2 + gy ** 2)
print(f"|gx| 平均 {np.abs(gx).mean():.3f}；|gy| 平均 {np.abs(gy).mean():.3f}")
print(f"邊緣強度最大的 5% 像素門檻 {np.percentile(mag, 95):.1f}")
print(f"cv2.Sobel 和 filter2D(sobel_x) 一樣嗎？ "
      f"{np.allclose(gx, cv2.filter2D(clean.astype(np.float32), cv2.CV_32F, sobel_x, borderType=cv2.BORDER_REFLECT_101))}")
cv2.imwrite(str(OUT / "sobel-mag.png"), np.clip(mag / 4, 0, 255).astype(np.uint8))
