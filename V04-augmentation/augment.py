"""V04 資料增強：torchvision.transforms v2；哪些增強對瑕疵是危險的；增強只用在訓練集。

    python V04-augmentation/augment.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).parent))
from chirality import handedness  # noqa: E402
from common import mvtec_paths  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(2)
OUT = Path("/tmp/ntunhs")
OUT.mkdir(parents=True, exist_ok=True)


def to_tensor(path, size=256):
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return tv_tensors.Image(torch.from_numpy(rgb).permute(2, 0, 1))


def to_gray(t):
    return cv2.cvtColor(t.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)


train_tf = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.3, contrast=0.3),
    v2.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
])
eval_tf = v2.Compose([v2.Resize(224, antialias=True)])

P = mvtec_paths("screw")
x = to_tensor(P["train"][0])
print("== 1. 同一張良品螺絲，訓練用增強跑 6 次 ==")
rows = []
for i in range(6):
    a = train_tf(x)
    g = to_gray(a)
    rows.append({"第幾次": i, "shape": tuple(a.shape), "平均亮度": g.mean(), "對比（標準差）": g.std()})
    cv2.imwrite(str(OUT / f"V04-aug-{i}.png"), cv2.cvtColor(a.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
print(pd.DataFrame(rows).set_index("第幾次").round(1).to_string())
e1, e2 = eval_tf(x), eval_tf(x)
print(f"評估用前處理跑兩次，結果相同：{torch.equal(e1, e2)}")

print("\n== 2. 影像和遮罩要一起轉（v2 認得 tv_tensors.Mask）==")
B = mvtec_paths("bottle")
k = next(i for i, m in enumerate(B["mask"]) if m)
img = to_tensor(B["test"][k])
mask = tv_tensors.Mask(torch.from_numpy(cv2.resize(cv2.imread(B["mask"][k], 0), (256, 256)) > 127))
geo = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation(degrees=30)])


def iou(a, b):
    a, b = a.numpy().astype(bool), b.numpy().astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1)


same, ious_bad = [], []
for seed in range(20):
    torch.manual_seed(seed)
    img2, mask2 = geo(img, mask)          # 對：同一次呼叫，影像和遮罩用同一組隨機參數
    torch.manual_seed(seed)
    img3 = geo(img)                       # 錯：影像一次……（同一個 seed，所以 img3 和 img2 一樣）
    mask3 = geo(mask)                     # ……遮罩又一次，抽到另一組參數
    same.append(torch.equal(img2, img3))
    ious_bad.append(iou(mask3, mask2))    # mask2 才是 img3 真正的瑕疵位置
print(f"瑕疵遮罩 {int(mask.sum())} 個像素；隨機翻轉＋旋轉 ±30 度，各試 20 次")
print(f"同一 seed 下，影像單獨轉和成對轉的結果相同：{all(same)}（所以 mask2 就是瑕疵真正的位置）")
print(f"分開轉的遮罩與真正位置的 IoU 平均 {np.mean(ious_bad):.3f}（最低 {np.min(ious_bad):.3f}）")

print("\n== 3. 危險的增強：metal_nut 的「翻面」本身就是瑕疵 ==")
N = mvtec_paths("metal_nut")
good = [cv2.imread(p, 0) for p in N["train"][:40]]
rows = [{"組": "train 良品 40 張", "右旋（>0）": sum(handedness(g) > 0 for g in good), "左旋（<0）": sum(handedness(g) < 0 for g in good)},
        {"組": "同 40 張水平翻轉後", "右旋（>0）": sum(handedness(cv2.flip(g, 1)) > 0 for g in good),
         "左旋（<0）": sum(handedness(cv2.flip(g, 1)) < 0 for g in good)}]
for d in ["good", "flip"]:
    gs = [cv2.imread(p, 0) for p, t in zip(N["test"], N["defect"]) if t == d]
    rows.append({"組": f"test {d} {len(gs)} 張", "右旋（>0）": sum(handedness(g) > 0 for g in gs),
                 "左旋（<0）": sum(handedness(g) < 0 for g in gs)})
print(pd.DataFrame(rows).set_index("組").to_string())

print("\n== 4. 增強只用在訓練集：先增強再切割會洩漏 ==")
# 任務：metal_nut 測試集 115 張，1-NN 分良品／瑕疵（只為了示範切割順序，不是好模型）
thumb = lambda t: cv2.resize(to_gray(t), (32, 32), interpolation=cv2.INTER_AREA).astype(float).ravel()  # noqa: E731
imgs = [to_tensor(p) for p in N["test"]]
y = N["y"]
aug = v2.Compose([v2.RandomRotation(degrees=5), v2.ColorJitter(brightness=0.2)])
COPIES = 4


def one_nn(Xtr, ytr, Xte, yte):
    d = ((Xte[:, None] - Xtr[None]) ** 2).sum(-1)
    return (ytr[d.argmin(1)] == yte).mean()


torch.manual_seed(0)
rng = np.random.default_rng(0)
# 錯：每張先做 4 個增強版，再隨機切一半當測試
X = np.array([thumb(aug(t)) for t in imgs for _ in range(COPIES)])
Y = np.repeat(y, COPIES)
order = rng.permutation(len(X))
a, b = order[:len(X) // 2], order[len(X) // 2:]
acc_leak = one_nn(X[a], Y[a], X[b], Y[b])
# 對：先切「原圖」，只對訓練那一半做增強
order = rng.permutation(len(imgs))
tr_i, te_i = order[:len(imgs) // 2], order[len(imgs) // 2:]
Xtr = np.array([thumb(aug(imgs[i])) for i in tr_i for _ in range(COPIES)])
Ytr = np.repeat(y[tr_i], COPIES)
Xte = np.array([thumb(imgs[i]) for i in te_i])
acc_ok = one_nn(Xtr, Ytr, Xte, y[te_i])
print(f"先增強再切：測試 {len(b)} 筆，1-NN 正確率 {acc_leak:.3f}")
print(f"先切再增強：測試 {len(te_i)} 張，1-NN 正確率 {acc_ok:.3f}")
print(f"只猜多數類（瑕疵）的正確率 {y[te_i].mean():.3f}")
