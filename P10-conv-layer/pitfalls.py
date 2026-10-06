"""P10｜讓 AI 寫，然後挑錯：HWC 直接丟進 Conv2d、忘了 padding、uint8 tensor、權重形狀少一維。"""
import cv2
import numpy as np
import torch
from torch import nn

from common import PHOTOS

torch.set_num_threads(4)
img = cv2.imread(str(PHOTOS / "bolts-conveyor.png"))[400:1000, 300:1100]
sobel_x = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])

print("== 錯誤 1：把 (高, 寬, 3) 直接丟進 Conv2d(3, ...) ==")
conv = nn.Conv2d(3, 8, 3, padding=1)
hwc = torch.from_numpy(img).float()[None]                   # (1, 600, 800, 3)
try:
    conv(hwc)
except RuntimeError as e:
    print("報錯：", " ".join(str(e).split())[:110])
chw = torch.from_numpy(img).permute(2, 0, 1).float()[None]
print("permute(2, 0, 1) 之後", tuple(chw.shape), "→ 輸出", tuple(conv(chw).shape))

print("\n== 錯誤 2：用 reshape 代替 permute ==")
bad = torch.from_numpy(img).float().reshape(1, 3, 600, 800)
print("reshape 也是 (1, 3, 600, 800)，不會報錯")
print(f"但「第 0 通道」的前 6 個數：reshape {bad[0, 0, 0, :6].int().tolist()}；"
      f"permute {chw[0, 0, 0, :6].int().tolist()}")
print("reshape 只是把一長串數字重新切，B、G、R 被混在同一個通道裡")

print("\n== 錯誤 3：忘了 padding，輸出變小一圈 ==")
g = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).float()[None, None]
for pad in [0, 1]:
    c = nn.Conv2d(1, 1, 3, padding=pad, bias=False)
    print(f"padding={pad}：輸入 {tuple(g.shape[2:])} → 輸出 {tuple(c(g).shape[2:])}")
print("疊 10 層 3×3 都沒 padding：每邊少 10 格 →", (600 - 20, 800 - 20))

print("\n== 錯誤 4：uint8 tensor 直接進 Conv2d ==")
try:
    nn.Conv2d(1, 1, 3)(torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[None, None])
except RuntimeError as e:
    print("報錯：", str(e).strip().splitlines()[0][:70])
print("要先 .float()；預訓練模型還要 /255 再減平均、除標準差（P11）")

print("\n== 錯誤 5：weight 指派成 (3, 3)，少了兩個維度 ==")
c = nn.Conv2d(1, 1, 3, bias=False)
print("conv.weight.shape", tuple(c.weight.shape))
try:
    c.weight = nn.Parameter(sobel_x)
    c(g)
except RuntimeError as e:
    print("報錯：", str(e).split(",")[0][:70])
c.weight = nn.Parameter(sobel_x[None, None])
print("sobel_x[None, None] →", tuple(c.weight.shape), "→ 可以算")
