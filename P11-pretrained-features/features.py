"""P11｜ResNet18（ImageNet）整張圖 → 512 維特徵向量（fc 前的 avgpool 輸出），有磁碟快取。

    from features import load_model, preprocess, resnet_vectors
    v = resnet_vectors(paths)      # (n, 512) float32
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
import torchvision
from PIL import Image

from common import CACHE

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
WEIGHTS = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),                                     # 0–255 → 0–1，HWC → CHW
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
_model = None


def load_model():
    """完整的 ResNet18（含最後的 1000 類分類層）。"""
    global _model
    if _model is None:
        _model = torchvision.models.resnet18(weights=WEIGHTS).eval()
    return _model


def resnet_vectors(paths, batch=16):
    """每張圖一個 512 維向量：把最後的分類層換成「什麼都不做」，拿到它前一層的輸出。"""
    key = hashlib.sha256("\n".join(map(str, paths)).encode()).hexdigest()[:16]
    f = CACHE / f"r18-avg512-{key}.npy"
    if f.exists():
        return np.load(f)
    model = load_model()
    fc, model.fc = model.fc, torch.nn.Identity()
    out = []
    try:
        with torch.no_grad():
            for i in range(0, len(paths), batch):
                x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
                out.append(model(x).numpy())
    finally:
        model.fc = fc
    arr = np.concatenate(out).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, arr)
    return arr
