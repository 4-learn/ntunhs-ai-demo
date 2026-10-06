"""V24–V26 共用：ResNet18 小格特徵抽取器（與 common.resnet_patch_features 同設定）的 ONNX 匯出、
不靠 torch 的前處理、onnxruntime session、延遲量測。

V25、V26、V28 用法：
    import sys; sys.path.insert(0, "V24-onnx-export")
    from edge_lib import ...

ONNX 檔存在 demo/cache/onnx/（.gitignore 排除 *.onnx，不進 repo；第一次執行會自動匯出）。
"""
from __future__ import annotations

import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from common import CACHE, MODELS

ONNX_DIR = CACHE / "onnx"
R18_ONNX = ONNX_DIR / "r18-l23.onnx"
POSE_ONNX = ONNX_DIR / "yolo11n-pose.onnx"
OPSET = 17
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- 模型（PyTorch 端）

def r18_patch_model():
    """回傳一個 nn.Module：輸入正規化後的影像 (n, 3, H, W)，輸出 (n, 384, H/8, W/8)。

    和 common.resnet_patch_features 完全同一套：ResNet18（ImageNet）layer2＋layer3（放大到 layer2 大小）
    → 3×3 平均。差別只在 common 用 hook 拿中間層，這裡把「只算到 layer3」寫成一個完整的模型，
    才匯出得成一個獨立的檔案（layer4 與分類層根本不用算）。
    """
    import torch
    import torchvision

    class R18Patch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
            self.layer2, self.layer3 = m.layer2, m.layer3

        def forward(self, x):
            l2 = self.layer2(self.stem(x))
            l3 = self.layer3(l2)
            l3 = torch.nn.functional.interpolate(l3, size=l2.shape[-2:], mode="bilinear")
            return torch.nn.functional.avg_pool2d(torch.cat([l2, l3], 1), 3, 1, 1)

    return R18Patch().eval()


# ---------------------------------------------------------------- 前處理（不用 torch，Pi 5 上也能跑）

def preprocess(paths, size=224) -> np.ndarray:
    """路徑 → (n, 3, size, size) float32。與 common 的 Resize＋ToTensor＋Normalize 相同（雙線性縮放）。"""
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        a = (np.asarray(im, np.float32) / 255 - MEAN) / STD
        out.append(a.transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(out), dtype=np.float32)


# ---------------------------------------------------------------- 匯出

def export_r18(path: Path = R18_ONNX, dynamic=True, force=False) -> Path:
    """torch.onnx.export：opset 17、batch 與高寬都是動態維度。已存在就不重匯。"""
    if path.exists() and not force:
        return path
    import torch

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    path.parent.mkdir(parents=True, exist_ok=True)
    axes = {"image": {0: "batch", 2: "height", 3: "width"},
            "patches": {0: "batch", 2: "h8", 3: "w8"}} if dynamic else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(r18_patch_model(), (torch.zeros(1, 3, 224, 224),), str(path),
                          dynamo=False,                      # 舊的 TorchScript 匯出器；新版 dynamo 匯出器要 onnxscript
                          opset_version=OPSET, input_names=["image"], output_names=["patches"],
                          dynamic_axes=axes)
    return path


def export_pose(path: Path = POSE_ONNX, force=False) -> Path:
    """ultralytics model.export(format="onnx")：離線可用（不需下載），simplify=False 因為沒裝 onnxslim。

    ultralytics 會把 .onnx 寫在 .pt 旁邊，所以先把 .pt 複製到 cache/onnx/，不弄髒 models/。
    在多人共用的教學機上要好幾分鐘（教師紀錄），之後直接讀快取。
    """
    if path.exists() and not force:
        return path
    os.environ.setdefault("YOLO_OFFLINE", "1")
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    path.parent.mkdir(parents=True, exist_ok=True)
    pt = path.with_suffix(".pt")
    shutil.copy(MODELS / "yolo11n-pose.pt", pt)
    out = YOLO(str(pt)).export(format="onnx", imgsz=640, dynamic=True, opset=OPSET, simplify=False, verbose=False)
    return Path(out)


# ---------------------------------------------------------------- 推論

def ort_session(path, threads=4):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_patches(sess, x: np.ndarray, batch=16) -> np.ndarray:
    """(n, 3, H, W) → (n, H/8, W/8, 384)，排列與 common.resnet_patch_features 相同。"""
    name = sess.get_inputs()[0].name
    out = [sess.run(None, {name: x[i:i + batch]})[0] for i in range(0, len(x), batch)]
    return np.concatenate(out).transpose(0, 2, 3, 1)


def p50_ms(fn, n=30, warmup=5) -> float:
    """跑 warmup 次不計，再跑 n 次，回傳延遲中位數（毫秒）。中位數比平均不怕偶發的卡頓。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)


def rel_order(a, b) -> str:
    """相對誤差（最大絕對差 ÷ 參考輸出的最大絕對值）的量級，例如 3.7e-04 → '< 1e-03'。

    只印量級，因為最後幾位小數會因 CPU 型號、執行緒數而不同，印精確值講義會對不上。
    """
    import math

    r = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
    return "0" if r == 0 else f"< 1e-{-math.ceil(math.log10(r)):02d}"


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def mb(path) -> float:
    return Path(path).stat().st_size / 1e6


# ---------------------------------------------------------------- 小子集的異常分數（V25 量化、V26 解析度）

N_TRAIN = 60  # 每類只取 train 前 60 張（換解析度要重算特徵，全量太慢、太吃記憶體）


def subset(cat, n_train=N_TRAIN):
    """回傳 (train 前 n_train 張, test 全部, y)。記憶／校正再用 memory_calib_split 在這 n_train 張上切。"""
    from common import mvtec_paths

    P = mvtec_paths(cat)
    return P["train"][:n_train], P["test"], P["y"]


def patch_scores(train_feats, test_feats):
    """小格 kNN（V10）：記憶 80% 的小格抽 10%（rng 0）當記憶庫；回傳 (校正分數, 測試分數)。"""
    sys_path_v10()
    from anomaly_lib import build_bank, patch_maps, split_index

    mem, cal = split_index(len(train_feats))
    bank = build_bank(train_feats, mem)
    cal_s = patch_maps(train_feats[cal], bank).reshape(len(cal), -1).max(1)
    test_s = patch_maps(test_feats, bank).reshape(len(test_feats), -1).max(1)
    return cal_s, test_s


def sys_path_v10():
    """anomaly_lib（V08–V10 共用的異常分數）放在 V08-anomaly-intro/；舊版在 V10-patch-knn/。"""
    import sys

    for d in ["V08-anomaly-intro", "V10-patch-knn"]:
        p = Path(__file__).resolve().parents[1] / d
        if (p / "anomaly_lib.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return


def onnx_features(model_path, paths, size=224, threads=4) -> np.ndarray:
    """用 ONNX 模型抽 (n, size/8, size/8, 384) 小格特徵；存成 float16 快取（鍵＝模型檔名＋路徑清單＋size）。"""
    import hashlib

    key = hashlib.sha256(("\n".join(map(str, paths)) + f"|{size}|{Path(model_path).name}").encode()).hexdigest()[:16]
    f = ONNX_DIR / f"feat-{Path(model_path).stem}-{size}-{key}.npy"
    if f.exists():
        return np.load(f).astype(np.float32)
    sess = ort_session(model_path, threads)
    out = [ort_patches(sess, preprocess(paths[i:i + 16], size)).astype(np.float16) for i in range(0, len(paths), 16)]
    arr = np.concatenate(out)
    np.save(f, arr)
    return arr.astype(np.float32)


def subset_scores(cat, size=224, model_path=None):
    """小子集（train 前 60 張、test 全部）在某解析度下的小格 kNN 分數；存快取。

    回傳 dict(calib, test, y, bank_n)。V26 的取捨表與 Workshop 共用。
    """
    model_path = Path(model_path or R18_ONNX)
    train, test, y = subset(cat)
    f = ONNX_DIR / f"scores-{model_path.stem}-{cat}-{size}-n{len(train)}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    ftr, fte = onnx_features(model_path, train, size), onnx_features(model_path, test, size)
    sys_path_v10()
    from anomaly_lib import build_bank, split_index

    bank_n = len(build_bank(ftr, split_index(len(ftr))[0]))
    cal, s = patch_scores(ftr, fte)
    out = dict(calib=cal, test=s, y=np.asarray(y), bank_n=np.array(bank_n))
    np.savez(f, **out)
    return out
