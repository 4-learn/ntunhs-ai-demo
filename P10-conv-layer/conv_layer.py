"""P10｜從手寫濾波器到卷積層：Sobel 核塞進 nn.Conv2d、ResNet18 conv1 的 64 個核、參數量計算。"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from common import PHOTOS

torch.set_num_threads(4)
torch.manual_seed(0)
OUT = Path("/tmp/ntunhs/P10")
OUT.mkdir(parents=True, exist_ok=True)

gray = cv2.cvtColor(cv2.imread(str(PHOTOS / "bolts-conveyor.png")), cv2.COLOR_BGR2GRAY)
roi = gray[400:1000, 300:1100].astype(np.float32)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)

print("== 1. nn.Conv2d 就是一組可以學的濾波核 ==")
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
print(conv)
print("weight shape", tuple(conv.weight.shape), "＝(輸出通道, 輸入通道, 高, 寬)")
print("剛建立時是隨機數（第一列）：", [round(float(v), 3) for v in conv.weight.detach()[0, 0, 0]])

print("\n== 2. 把 P06 的 Sobel 核放進去 ==")
with torch.no_grad():
    conv.weight[:] = torch.from_numpy(sobel_x)[None, None]
x = torch.from_numpy(roi)[None, None]                # (批次, 通道, 高, 寬) = (1, 1, 600, 800)
print("輸入 tensor shape", tuple(x.shape))
with torch.no_grad():
    y = conv(x)[0, 0].numpy()
ref = cv2.filter2D(roi, cv2.CV_32F, sobel_x, borderType=cv2.BORDER_CONSTANT)   # 補 0，和 padding=1 一樣
print("輸出 shape", y.shape, "；cv2.filter2D shape", ref.shape)
print(f"最大差 {np.abs(y - ref).max():.6f}")
print("PyTorch 也不翻轉核（和 P06 的 cv2.filter2D 一樣是「相關」）")

print("\n== 3. 一次放 2 個核：輸出就有 2 個通道 ==")
conv2 = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
with torch.no_grad():
    conv2.weight[0, 0] = torch.from_numpy(sobel_x)
    conv2.weight[1, 0] = torch.from_numpy(sobel_x.T.copy())
    out = conv2(x)[0].numpy()
mag = np.sqrt(out[0] ** 2 + out[1] ** 2)
gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, borderType=cv2.BORDER_CONSTANT)
gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_CONSTANT)
print("輸出 shape", out.shape, "← 每個核產生一張「特徵圖」")
print(f"邊緣強度和 P06 的 sqrt(gx²+gy²) 最大差 {np.abs(mag - np.sqrt(gx ** 2 + gy ** 2)).max():.4f}")

print("\n== 4. 彩色輸入：一個核同時看 3 個通道 ==")
img = cv2.imread(str(PHOTOS / "bolts-conveyor.png"))[400:1000, 300:1100]
rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None]
print("彩色 tensor shape", tuple(rgb.shape), "← (1, 3, 高, 寬)：通道要搬到前面")
conv3 = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
with torch.no_grad():
    conv3.weight[0] = torch.from_numpy(sobel_x)[None].repeat(3, 1, 1) / 3   # 3 個通道各用 Sobel，平均
    o3 = conv3(rgb)[0, 0].numpy()
avg = rgb[0].mean(0).numpy()
print("weight shape", tuple(conv3.weight.shape), "：每個核是 3×3×3 = 27 個數")
o_avg = cv2.filter2D(avg, cv2.CV_32F, sobel_x, borderType=cv2.BORDER_CONSTANT)
print(f"等於「先把 RGB 平均成灰階，再做 Sobel」：最大差 {np.abs(o3 - o_avg).max():.4f}")

print("\n== 5. ResNet18 的第一層：64 個學出來的 7×7×3 核 ==")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
w = model.conv1.weight.detach().numpy()
print(model.conv1)
print("weight shape", w.shape)
gray_w = w.mean(1)                                       # 把 3 個顏色通道平均，看形狀
s = w.sum(axis=(1, 2, 3))
print(f"64 個核的總和：最小 {s.min():.3f}、中位數 {np.median(s):.3f}、最大 {s.max():.3f}")
print(f"總和接近 0（|總和| < 0.1）的核：{(np.abs(s) < 0.1).sum()} 個 ← 像 Sobel，只對「變化」有反應")
color_diff = np.abs(w - w.mean(1, keepdims=True)).mean(axis=(1, 2, 3)) / (np.abs(w).mean(axis=(1, 2, 3)) + 1e-8)
print(f"三個顏色通道差很多（顏色核，比例 > 0.5）：{(color_diff > 0.5).sum()} 個；"
      f"三通道幾乎一樣（亮度核，< 0.2）：{(color_diff < 0.2).sum()} 個")

print("\n== 6. 和 Sobel x 最像的學習核 ==")
sx7 = np.zeros((7, 7), np.float32)
sx7[2:5, 2:5] = sobel_x


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


sims = np.array([cos(gray_w[i], sx7) for i in range(64)])
sims_y = np.array([cos(gray_w[i], sx7.T) for i in range(64)])
best = np.argsort(-np.abs(sims))[:3]
for i in best:
    print(f"第 {i:>2} 號核：和 Sobel x 的相似度 {sims[i]:+.3f}；和 Sobel y {sims_y[i]:+.3f}")
i = int(best[0])
print(f"第 {i} 號核（3 通道平均後）中間 3×3：")
print(np.round(gray_w[i, 2:5, 2:5] / np.abs(gray_w[i, 2:5, 2:5]).max(), 2))
print("沒有人告訴它要找邊——它從 120 萬張照片裡自己學出「找邊」比較有用")
grid = np.zeros((8 * 9, 8 * 9, 3), np.float32)
for k in range(64):
    r, c = divmod(k, 8)
    t = w[k].transpose(1, 2, 0)
    grid[r * 9:r * 9 + 7, c * 9:c * 9 + 7] = (t - t.min()) / (t.max() - t.min() + 1e-8)
cv2.imwrite(str(OUT / "resnet18-conv1.png"),
            cv2.resize((grid[..., ::-1] * 255).astype(np.uint8), None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST))

print("\n== 7. 參數量：一層卷積有幾個數字要學 ==")


def n_params(m):
    return sum(p.numel() for p in m.parameters())


for name, layer in [("Conv2d(1, 1, 3, bias=False)", nn.Conv2d(1, 1, 3, bias=False)),
                    ("Conv2d(1, 2, 3, bias=False)", nn.Conv2d(1, 2, 3, bias=False)),
                    ("Conv2d(3, 1, 3, bias=False)", nn.Conv2d(3, 1, 3, bias=False)),
                    ("ResNet18 conv1 = Conv2d(3, 64, 7, bias=False)", model.conv1),
                    ("Conv2d(64, 64, 3)（有 bias）", nn.Conv2d(64, 64, 3))]:
    print(f"{name:<46} {n_params(layer):>7,}")
print("公式：輸出通道 × 輸入通道 × 核高 × 核寬（+ 輸出通道個 bias）")
print(f"ResNet18 全部 {n_params(model):,} 個參數；conv1 只占 {n_params(model.conv1) / n_params(model):.4f}")
fc = nn.Linear(3 * 224 * 224, 64)
print(f"如果第一層改用「全連接」接 224×224×3 輸入、輸出 64 個數：{n_params(fc):,} 個參數")
print(f"是 conv1 的 {n_params(fc) / n_params(model.conv1):,.0f} 倍——卷積「同一個核滑過整張圖」省下來的")
