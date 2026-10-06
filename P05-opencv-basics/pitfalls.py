"""P05｜讓 AI 寫，然後挑錯：resize 寬高顛倒、RGB2HSV 用在 BGR、H 用 0–360 度、INTER_NEAREST 縮圖。"""
import cv2
import numpy as np

from common import PHOTOS

room = cv2.imread(str(PHOTOS / "care-room.png"))

print("== 錯誤 1：cv2.resize(img, (240, 320)) ==")
ai = cv2.resize(room, (240, 320))
ok = cv2.resize(room, (320, 240), interpolation=cv2.INTER_AREA)
print(f"AI 版 shape {ai.shape}（直式、人被壓扁）；正確版 {ok.shape}")
print(f"AI 版寬高比 {ai.shape[1] / ai.shape[0]:.3f}；原圖 {room.shape[1] / room.shape[0]:.3f}")

print("\n== 錯誤 2：H 用 0–360 度、又用 RGB2HSV ==")
# 題目：找出木頭家具（橙褐色，色相約 16–44 度）
ai_mask = cv2.inRange(cv2.cvtColor(room, cv2.COLOR_RGB2HSV), (16, 80, 60), (44, 255, 230))
half = cv2.inRange(cv2.cvtColor(room, cv2.COLOR_RGB2HSV), (8, 80, 60), (22, 255, 230))
deg = cv2.inRange(cv2.cvtColor(room, cv2.COLOR_BGR2HSV), (16, 80, 60), (44, 255, 230))
right = cv2.inRange(cv2.cvtColor(room, cv2.COLOR_BGR2HSV), (8, 80, 60), (22, 255, 230))
print(f"AI 版（RGB2HSV＋度數）         木頭色占 {(ai_mask > 0).mean():.3f}")
print(f"只修 H ÷ 2（仍是 RGB2HSV）     木頭色占 {(half > 0).mean():.3f}")
print(f"只修 BGR2HSV（H 仍是度數）     木頭色占 {(deg > 0).mean():.3f}")
print(f"兩個都修                       木頭色占 {(right > 0).mean():.3f}")

print("\n== 錯誤 3：縮成 1/4 用預設內插，螺紋變成鋸齒 ==")
bolts = cv2.cvtColor(cv2.imread(str(PHOTOS / "bolts-conveyor.png")), cv2.COLOR_BGR2GRAY)[:1084]  # 高裁成 4 的倍數
h, w = bolts.shape
for name, flag in [("INTER_NEAREST", cv2.INTER_NEAREST), ("INTER_LINEAR（預設）", cv2.INTER_LINEAR),
                   ("INTER_AREA", cv2.INTER_AREA)]:
    q = cv2.resize(bolts, (w // 4, h // 4), interpolation=flag)
    ref = bolts.reshape(h // 4, 4, w // 4, 4).mean((1, 3))    # 4×4 平均＝理想縮圖
    print(f"{name:<18} 與 4×4 平均的差 {np.abs(q - ref).mean():.3f}")
