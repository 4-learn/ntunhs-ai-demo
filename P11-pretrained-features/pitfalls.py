"""P11｜讓 AI 寫，然後挑錯：記憶良品和查詢圖前處理不一致（沒標準化、cv2 的 BGR、忘了 eval）。"""
import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import mvtec_paths                       # noqa: E402
from features import load_model, preprocess, resnet_vectors   # noqa: E402

B = mvtec_paths("bottle")
memory = resnet_vectors(B["train"])                  # 記憶良品：用正確的前處理算好、存在快取
idx = [i for i, d in enumerate(B["defect"]) if d == "good"][:8] + \
      [i for i, d in enumerate(B["defect"]) if d == "broken_large"][:8]
paths = [B["test"][i] for i in idx]
y = np.array([0] * 8 + [1] * 8)

no_norm = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                          torchvision.transforms.ToTensor()])


def bgr_tensor(p):
    img = cv2.resize(cv2.imread(p), (224, 224))      # cv2：BGR、沒換成 RGB
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255
    return torchvision.transforms.functional.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def vectors(make, train_mode=False):
    model = copy.deepcopy(load_model())              # 複製一份，train 模式不會弄髒共用的模型
    model.fc = torch.nn.Identity()
    model.train(train_mode)
    with torch.no_grad():
        return model(torch.stack([make(p) for p in paths])).numpy()


def report(name, v):
    d = np.sqrt(((v[:, None] - memory[None]) ** 2).sum(-1)).min(1)
    sep = d[y == 1].min() > d[y == 0].max()
    print(f"{name:<22} 良品最近距離 {d[y == 0].mean():6.2f}  瑕疵 {d[y == 1].mean():6.2f}  "
          f"一刀分得開：{sep}")


print("== 記憶良品用正確前處理；查詢的 8 張良品＋8 張 broken_large 用不同前處理 ==")
right = vectors(lambda p: preprocess(Image.open(p).convert("RGB")))
report("正確（和記憶一致）", right)

print("\n== 錯誤 1：查詢時沒做 Normalize ==")
report("沒標準化", vectors(lambda p: no_norm(Image.open(p).convert("RGB"))))

print("\n== 錯誤 2：查詢時用 cv2.imread（BGR）==")
v = vectors(bgr_tensor)
report("BGR 當 RGB", v)
print(f"同一張圖 BGR 與 RGB 的向量距離 {np.linalg.norm(v - right, axis=1).mean():.2f}")
print("瓶子幾乎是灰色，還分得開；但良品的距離翻倍，門檻（V12）會全部失準")

print("\n== 錯誤 3：忘了 model.eval() ==")
v = vectors(lambda p: preprocess(Image.open(p).convert("RGB")), train_mode=True)
report("train 模式", v)
print("train 模式的 BatchNorm 用「這一批」的平均：同一張圖放在不同批，向量就不同")
