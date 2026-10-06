"""P06｜讓 AI 寫，然後挑錯：uint8 輸出吃掉負的邊、gx+gy 互相抵消、核沒有正規化。"""
import cv2
import numpy as np

from common import PHOTOS

gray = cv2.cvtColor(cv2.imread(str(PHOTOS / "bolts-conveyor.png")), cv2.COLOR_BGR2GRAY)
roi = gray[400:1000, 300:1100]
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)

print("== 錯誤 1：cv2.filter2D(gray, -1, sobel_x)，輸出還是 uint8 ==")
ai = cv2.filter2D(roi, -1, sobel_x)                         # -1 = 跟輸入一樣是 uint8
ok = cv2.filter2D(roi, cv2.CV_32F, sobel_x)
strong = np.abs(ok) > 100
print(f"正確（float32）值域 {ok.min():.0f} ~ {ok.max():.0f}")
print(f"AI 版（uint8）值域 {ai.min()} ~ {ai.max()}")
kept = (ai[strong] > 100).mean()
print(f"強邊緣（|值|>100）中，AI 版還看得到的比例 {kept:.3f}")
print("→ 「由亮變暗」的邊是負數，被截成 0；螺栓只剩一側有邊")

print("\n== 錯誤 2：邊緣強度寫成 gx + gy ==")
gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)
mag = np.sqrt(gx ** 2 + gy ** 2)
bad = gx + gy
print(f"sqrt(gx²+gy²) 平均 {mag.mean():.3f}")
print(f"gx + gy        平均 {bad.mean():.3f}（正負抵消）；|gx + gy| 平均 {np.abs(bad).mean():.3f}")
diag = (np.abs(bad) < 0.2 * mag) & (mag > 100)
print(f"強邊緣中，|gx+gy| 不到真實強度 2 成的比例 {diag.sum() / (mag > 100).sum():.3f}")

print("\n== 錯誤 3：自己做 5×5 平均核，忘了除以 25 ==")
k_bad = np.ones((5, 5), np.float32)
k_ok = k_bad / 25
b1 = cv2.filter2D(roi, -1, k_bad)
b2 = cv2.filter2D(roi, -1, k_ok)
print(f"沒除 25：平均亮度 {b1.mean():.3f}，被截成 255 的像素占 {(b1 == 255).mean():.3f}")
print(f"有除 25：平均亮度 {b2.mean():.3f}，原圖 {roi.mean():.3f}")
print(f"核的總和 = 亮度放大倍數：{k_bad.sum():.0f} vs {k_ok.sum():.0f}")
