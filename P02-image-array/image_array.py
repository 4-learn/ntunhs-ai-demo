"""P02｜影像就是陣列：shape、dtype、通道順序、切片裁切、亮度統計、uint8 溢位。"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import PHOTOS

OUT = Path("/tmp/ntunhs/P02")
OUT.mkdir(parents=True, exist_ok=True)

print("== 1. 一張照片 = 一個 3 維陣列 ==")
img = cv2.imread(str(PHOTOS / "bolts-conveyor.png"))
print(f"type {type(img).__name__}  shape {img.shape}  dtype {img.dtype}")
h, w, c = img.shape
print(f"高 {h} × 寬 {w} × {c} 個通道 = {img.size} 個數字，{img.nbytes / 1e6:.1f} MB")
print(f"最小 {img.min()}、最大 {img.max()}（uint8 只能放 0–255）")

print("\n== 2. 通道順序：OpenCV 是 BGR，PIL／matplotlib 是 RGB ==")
rgb = np.array(Image.open(PHOTOS / "bolts-conveyor.png").convert("RGB"))
print("cv2  左上角像素", img[0, 0].tolist(), "（B, G, R）")
print("PIL  左上角像素", rgb[0, 0].tolist(), "（R, G, B）")
print("img[..., ::-1] 和 PIL 完全相同：", np.array_equal(img[..., ::-1], rgb))
print("cv2.cvtColor(BGR2RGB) 也相同：", np.array_equal(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rgb))

print("\n== 3. 切片就是裁切：先列（y）後欄（x） ==")
y1, y2, x1, x2 = 140, 430, 750, 940            # 彎曲的那支螺栓
crop = img[y1:y2, x1:x2]
print(f"img[{y1}:{y2}, {x1}:{x2}] → shape {crop.shape}")
cv2.imwrite(str(OUT / "bent-bolt.png"), crop)
small = img[::4, ::4]
print(f"每 4 格取 1 格 img[::4, ::4] → {small.shape}（{small.size / img.size:.4f} 的資料量）")
print("切片是 view，不是複本：np.shares_memory =", np.shares_memory(crop, img))
mark = img.copy()
mark[y1:y2, x1:x1 + 3] = (0, 0, 255)           # 在複本上畫一條紅線（BGR 的紅）
print("改複本，原圖不變：", np.array_equal(img[y1:y2, x1], mark[y1:y2, x1]) is False)

print("\n== 4. 亮度統計：打光不均看得到數字 ==")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"灰階 shape {gray.shape}；平均亮度 {gray.mean():.3f}")
print("把畫面切成 3×3，每格平均亮度：")
for i in range(3):
    row = [gray[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3].mean() for j in range(3)]
    print("   " + "  ".join(f"{v:6.1f}" for v in row))
night = cv2.cvtColor(cv2.imread(str(PHOTOS / "care-corridor-fall.png")), cv2.COLOR_BGR2GRAY)
p1, p50, p99 = np.percentile(night, [1, 50, 99])
print(f"夜間走廊：平均 {night.mean():.3f}；1% / 50% / 99% 分位數 {p1:.0f} / {p50:.0f} / {p99:.0f}")

print("\n== 5. uint8 溢位：250 + 10 不是 260 ==")
px = np.array([[100, 200, 250]], dtype=np.uint8)  # 1 列 3 格的小圖
print("px          ", px.ravel().tolist())
print("px + 10     ", (px + np.uint8(10)).ravel().tolist(), "← 250 繞回 4")
print("px - 210    ", (px - np.uint8(210)).ravel().tolist(), "← 負數也繞回去")
print("cv2.add     ", cv2.add(px, np.full_like(px, 10)).ravel().tolist(), "← 封頂在 255")
safe = np.clip(px.astype(np.int16) + 10, 0, 255).astype(np.uint8)
print("先轉 int16  ", safe.ravel().tolist())

print("\n== 6. 夜間走廊太暗，整張亮度 ×3 ==")
bad = night * np.uint8(3)
ok = np.clip(night.astype(np.int16) * 3, 0, 255).astype(np.uint8)
print(f"原圖平均 {night.mean():.3f}；原本亮度 > 85 的像素占 {(night > 85).mean():.3f}")
print(f"night * 3（溢位）平均 {bad.mean():.3f}；有 {(bad < night).mean():.3f} 的像素反而比原圖暗")
print(f"先轉 int16 再 clip 平均 {ok.mean():.3f}；和 cv2.multiply 相同：{np.array_equal(ok, cv2.multiply(night, np.full_like(night, 3)))}")
cv2.imwrite(str(OUT / "night-x3-bad.png"), bad)
cv2.imwrite(str(OUT / "night-x3-ok.png"), ok)
