"""P05｜OpenCV 入門：讀圖、BGR／RGB、灰階、HSV、縮放、裁切。"""
from pathlib import Path

import cv2
import numpy as np

from common import PHOTOS

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # 讀不到檔的警告不印（講義輸出要穩定）

OUT = Path("/tmp/ntunhs/P05")
OUT.mkdir(parents=True, exist_ok=True)

print("== 1. 讀圖的三種結果 ==")
room = cv2.imread(str(PHOTOS / "care-room.png"))
gray_direct = cv2.imread(str(PHOTOS / "care-room.png"), cv2.IMREAD_GRAYSCALE)
missing = cv2.imread(str(PHOTOS / "care-room.PNG.bak"))
print("彩色 IMREAD_COLOR     ", room.shape, room.dtype)
print("灰階 IMREAD_GRAYSCALE ", gray_direct.shape, gray_direct.dtype)
print("檔名打錯             ", missing, "← 不報錯，回傳 None")

print("\n== 2. 同一個像素，BGR 和 RGB ==")
y, x = 200, 750                                   # 衣櫃木門
rgb = cv2.cvtColor(room, cv2.COLOR_BGR2RGB)
print(f"衣櫃 (y={y}, x={x})  BGR {room[y, x].tolist()}  RGB {rgb[y, x].tolist()}")
print("R 最大的像素比例：BGR 陣列當 RGB 讀", f"{(room[..., 0] > room[..., 2]).mean():.3f}",
      "；正確", f"{(rgb[..., 0] > rgb[..., 2]).mean():.3f}")
cv2.imwrite(str(OUT / "room-swapped.png"), rgb)    # 故意把 RGB 交給 imwrite：存出來木頭變藍色

print("\n== 3. 灰階：一個數字代表亮度 ==")
gray = cv2.cvtColor(room, cv2.COLOR_BGR2GRAY)
b, g, r = [room[..., i].astype(np.float32) for i in range(3)]
manual = np.round(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
avg3 = room.mean(axis=2)
print("cvtColor 灰階與 0.299R+0.587G+0.114B 最大差", int(np.abs(gray.astype(int) - manual).max()))
print(f"三通道直接平均 vs 正確灰階：平均差 {np.abs(avg3 - gray).mean():.3f}")
d = np.abs(gray_direct.astype(int) - gray)
print(f"IMREAD_GRAYSCALE 與 cvtColor：最大差 {d.max()}，有差的像素占 {(d > 0).mean():.3f}")

print("\n== 4. HSV：把「什麼顏色」和「多亮」分開 ==")
hsv = cv2.cvtColor(room, cv2.COLOR_BGR2HSV)
for name, (yy, xx) in {"衣櫃木門": (200, 750), "地磚": (850, 600), "長者背心": (380, 980)}.items():
    print(f"{name:<6} HSV {hsv[yy, xx].tolist()}")
print("OpenCV 的 H 範圍 0–179（度數 ÷ 2）；S、V 範圍 0–255")
wood = cv2.inRange(hsv, (8, 80, 60), (22, 255, 230))
floor = cv2.inRange(hsv, (0, 0, 150), (179, 40, 255))
print(f"木頭色（H 8–22、S≥80）占畫面 {(wood > 0).mean():.3f}")
print(f"地磚（S≤40、V≥150）占畫面 {(floor > 0).mean():.3f}")
wrong = cv2.inRange(cv2.cvtColor(room, cv2.COLOR_RGB2HSV), (8, 80, 60), (22, 255, 230))
print(f"同一個門檻，但用 RGB2HSV 轉 BGR 陣列：木頭色只剩 {(wrong > 0).mean():.3f}")
cv2.imwrite(str(OUT / "wood-mask.png"), wood)

print("\n== 5. 螺栓照片：沒有顏色，只有亮度 ==")
bolts = cv2.imread(str(PHOTOS / "bolts-conveyor.png"))
hb = cv2.cvtColor(bolts, cv2.COLOR_BGR2HSV)
w3 = bolts.shape[1] // 3
for name, sl in {"左 1/3": slice(0, w3), "中 1/3": slice(w3, 2 * w3), "右 1/3": slice(2 * w3, None)}.items():
    print(f"{name}  平均 S {hb[:, sl, 1].mean():6.3f}   平均 V {hb[:, sl, 2].mean():7.3f}")
print("care-room 平均 S", f"{hsv[..., 1].mean():.3f}", "；bolts 平均 S", f"{hb[..., 1].mean():.3f}")

print("\n== 6. 縮放：cv2.resize 的尺寸是 (寬, 高) ==")
small = cv2.resize(room, (320, 240), interpolation=cv2.INTER_AREA)
print("cv2.resize(room, (320, 240)) →", small.shape, "＝EZCam 的解析度")
print(f"縮放倍率 寬 {320 / room.shape[1]:.3f}、高 {240 / room.shape[0]:.3f}")
box = (880, 270, 1065, 665)                        # 扶助行器站著的長者 (x1, y1, x2, y2)
s = 320 / room.shape[1]
print(f"站著的長者：原圖高 {box[3] - box[1]} 像素 → 320×240 只剩 {round((box[3] - box[1]) * s)} 像素")

print("\n== 7. 縮小時用哪種內插？（縮成 1/8 再放大回來比誤差）==")
g8 = cv2.cvtColor(bolts, cv2.COLOR_BGR2GRAY)
h, w = g8.shape
for name, flag in [("INTER_NEAREST", cv2.INTER_NEAREST), ("INTER_LINEAR", cv2.INTER_LINEAR),
                   ("INTER_AREA", cv2.INTER_AREA)]:
    tiny = cv2.resize(g8, (w // 8, h // 8), interpolation=flag)
    back = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_LINEAR)
    print(f"{name:<14} 平均誤差 {cv2.absdiff(back, g8).mean():.3f}")

print("\n== 8. 給模型的 224×224：拉伸、中央裁切、補邊 ==")
H, W = room.shape[:2]
stretch = cv2.resize(room, (224, 224), interpolation=cv2.INTER_AREA)
side = min(H, W)
y0, x0 = (H - side) // 2, (W - side) // 2
center = cv2.resize(room[y0:y0 + side, x0:x0 + side], (224, 224), interpolation=cv2.INTER_AREA)
scale = 224 / max(H, W)
nw, nh = round(W * scale), round(H * scale)
pad = np.zeros((224, 224, 3), np.uint8)
top = (224 - nh) // 2
pad[top:top + nh, :nw] = cv2.resize(room, (nw, nh), interpolation=cv2.INTER_AREA)
print(f"原圖寬高比 {W / H:.3f}")
print(f"拉伸：寬高比被改成 1.000，人物變瘦 {1 - (H / W):.3f}")
print(f"中央裁切：保留 {side * side / (H * W):.3f} 的畫面；裁掉左右各 {x0} 像素")
sit = (105, 375)                                   # 坐著的長者 x 範圍
kept = max(0, sit[1] - x0) / (sit[1] - sit[0])
print(f"  坐著的長者（x {sit[0]}–{sit[1]}）只剩 {kept:.3f} 在畫面裡")
print(f"補邊（letterbox）：內容縮成 {nw}×{nh}，上下各補 {top} 列黑邊")
for name, im in [("stretch", stretch), ("center", center), ("letterbox", pad)]:
    cv2.imwrite(str(OUT / f"224-{name}.png"), im)
