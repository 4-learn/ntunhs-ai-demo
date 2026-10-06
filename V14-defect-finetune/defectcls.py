"""V14｜共用小工具：metal_nut 良品／瑕疵切割、讀圖、MobileNetV3-Small 微調。

train.py、evaluate.py、pitfalls.py 與 Workshop 解答都 import 這支。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from common import MODELS, mvtec_paths

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))  # 機器多人共用：2 執行緒
WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1
NORMALIZE = T.Normalize(WEIGHTS.transforms().mean, WEIGHTS.transforms().std)
SIZE = 224
CAT = "metal_nut"
MODEL_FILE = MODELS / "defect-cls-mnv3s.pt"
META_FILE = MODELS / "defect-cls-mnv3s.json"
FN_COST, FP_COST = 20, 1  # 王課長：漏到客戶一次＝多一次人工複檢的 20 倍


def split_table(cat: str = CAT) -> pd.DataFrame:
    """一張圖一列：path、source（train_good／test）、defect、y（1＝瑕疵）、split。

    - MVTec test（有瑕疵、也有少數良品）：每個瑕疵類型各自 60/20/20 切 train/val/test，
      seed 0，和 V15 的 tools/build_yolo_defects.py 逐張相同（V18 才能比較同一批測試圖）。
    - MVTec train/good（只有良品）：同樣 60/20/20、seed 1，讓 val、test 有足夠的良品算誤報。
    """
    rows = []
    P = mvtec_paths(cat)
    rng = np.random.default_rng(0)
    groups: dict[str, list[int]] = {}
    for i, d in enumerate(P["defect"]):
        groups.setdefault(d, []).append(i)
    for d, idx in groups.items():
        idx = list(rng.permutation(idx))
        a, b = int(round(len(idx) * 0.6)), int(round(len(idx) * 0.8))
        for split, part in (("train", idx[:a]), ("val", idx[a:b]), ("test", idx[b:])):
            for i in part:
                rows.append(dict(path=P["test"][i], source="test", defect=d, y=int(d != "good"), split=split,
                                 name=f"{cat}-{d}-{i:03d}"))
    rng = np.random.default_rng(1)
    order = rng.permutation(len(P["train"]))
    a, b = int(round(len(order) * 0.6)), int(round(len(order) * 0.8))
    for split, part in (("train", order[:a]), ("val", order[a:b]), ("test", order[b:])):
        for i in part:
            rows.append(dict(path=P["train"][i], source="train_good", defect="good", y=0, split=split,
                             name=f"{cat}-train_good-{i:03d}"))
    return pd.DataFrame(rows)


def load_images(paths) -> torch.Tensor:
    """(N, 3, 224, 224)、0～1。"""
    return torch.stack([T.functional.to_tensor(Image.open(p).convert("RGB").resize((SIZE, SIZE)))
                        for p in paths])


def build_model(unfreeze_from: int | None = 9) -> nn.Module:
    """預訓練 MobileNetV3-Small，最後一層換成 2 類（良品／瑕疵）。

    unfreeze_from=None：只訓練最後一層（ML 課 17 節的做法）。
    unfreeze_from=9：features[9:]（最後幾個區塊，7×7 那一段）與分類頭一起訓練。
    """
    model = mobilenet_v3_small(weights=WEIGHTS)
    for p in model.parameters():
        p.requires_grad = False
    if unfreeze_from is not None:
        for p in model.features[unfreeze_from:].parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    model.classifier[3] = nn.Linear(1024, 2)
    return model


def n_params(model: nn.Module) -> tuple[int, int]:
    return (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad))


def frozen_front(model: nn.Module, X: torch.Tensor, split: int = 9, batch=16) -> torch.Tensor:
    """凍結的前段（features[:split]）只算一次。增強只用 4 種旋轉，所以 4 種都先算好：
    回傳 (4, N, C, H, W)。CPU 上省下大部分時間（ML 課 17 節「骨幹只算一次」的延伸）。"""
    front = model.features[:split].eval()
    out = []
    with torch.no_grad():
        for k in range(4):
            out.append(torch.cat([front(NORMALIZE(torch.rot90(X[i:i + batch], k, dims=(2, 3))))
                                  for i in range(0, len(X), batch)]))
    return torch.stack(out)


def tail_forward(model: nn.Module, F: torch.Tensor, split: int = 9) -> torch.Tensor:
    x = model.features[split:](F)
    return model.classifier(torch.flatten(model.avgpool(x), 1))


def train(model: nn.Module, F4: torch.Tensor, y, epochs=8, batch=8, lr=1e-3, seed=0, split: int = 9,
          log=None) -> list[float]:
    """完整的小訓練迴圈（zero_grad、前向、loss、backward、step）：只跑可訓練的後段。

    每個 epoch 每張圖隨機取一種旋轉（0/90/180/270 度），再打亂順序。
    瑕疵比良品少，用類別權重讓兩類的 loss 份量相當（ML 課 08 節 class_weight 的同一個想法）。
    """
    torch.manual_seed(seed)
    y = torch.tensor(np.asarray(y), dtype=torch.long)
    w = len(y) / (2 * torch.bincount(y, minlength=2).float())
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    history = []
    for ep in range(epochs):
        model.train()
        rot = torch.randint(0, 4, (len(y),), generator=g)
        order = torch.randperm(len(y), generator=g)
        total = 0.0
        for i in range(0, len(y), batch):
            j = order[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(tail_forward(model, F4[rot[j], j], split), y[j])
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        history.append(total / len(y))
        if log:
            log(ep + 1, history[-1])
    model.eval()
    return history


def prob_defect(model: nn.Module, X: torch.Tensor, batch=16) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(NORMALIZE(X[i:i + batch])).softmax(1)[:, 1]
                          for i in range(0, len(X), batch)]).numpy()


def load_trained() -> nn.Module:
    model = build_model(None)
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    return model.eval()


def scores(y, prob, thr) -> dict:
    y, pred = np.asarray(y), np.asarray(prob) >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                precision=round(tp / (tp + fp), 3) if tp + fp else 0.0,
                recall=round(tp / (tp + fn), 3) if tp + fn else 0.0,
                成本=FN_COST * fn + FP_COST * fp)


def pick_threshold(y, prob, grid=None) -> float:
    """在 val 上掃門檻，取 20×漏報＋1×誤報 最低者；同分取較大的門檻（少吵一點）。"""
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.05), 2)
    costs = [scores(y, prob, t)["成本"] for t in grid]
    best = min(costs)
    return float(max(t for t, c in zip(grid, costs) if c == best))
