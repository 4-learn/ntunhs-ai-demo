"""P02｜讓 AI 寫，然後挑錯：裁切 x/y 顛倒、BGR 當 RGB、uint8 平均差異。"""
import cv2
import numpy as np

from common import PHOTOS

img = cv2.imread(str(PHOTOS / "bolts-conveyor.png"))
x1, y1, x2, y2 = 750, 140, 940, 430           # 彎曲螺栓的框：左上 (x1, y1)、右下 (x2, y2)

print("== 錯誤 1：img[x1:x2, y1:y2] ==")
wrong = img[x1:x2, y1:y2]
right = img[y1:y2, x1:x2]
print(f"AI 版 shape {wrong.shape}；正確版 shape {right.shape}")
print(f"AI 版平均亮度 {wrong.mean():.3f}；正確版 {right.mean():.3f}（裁到的是另一塊輸送帶）")

print("\n== 錯誤 2：用 (255, 0, 0) 畫「紅框」 ==")
box = img.copy()
cv2.rectangle(box, (x1, y1), (x2, y2), (255, 0, 0), 3)
print("框線上的像素", box[y1, x1 + 50].tolist(), "→ 在 BGR 裡是藍色")
cv2.rectangle(box, (x1, y1), (x2, y2), (0, 0, 255), 3)
print("改用 (0, 0, 255)", box[y1, x1 + 50].tolist(), "→ 才是紅色")

print("\n== 錯誤 3：兩張圖相減找差異 ==")
a = img[y1:y2, x1:x2]
shifted = np.roll(a, 2, axis=1)               # 同一塊往右偏 2 像素，模擬相機抖動
d_bad = a - shifted
d_ok = cv2.absdiff(a, shifted)
print(f"uint8 直接相減：平均差 {d_bad.mean():.3f}、最大 {d_bad.max()}")
print(f"cv2.absdiff   ：平均差 {d_ok.mean():.3f}、最大 {d_ok.max()}")
print(f"先轉 int16 取絕對值：平均差 {np.abs(a.astype(np.int16) - shifted).mean():.3f}")
