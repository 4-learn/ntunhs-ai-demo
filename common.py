"""國北護課程（機器學習補充 P 節、影像 AI V 節）共用：路徑、資料讀取、切割、骨架特徵。

每一節的 demo 都從 repo 根目錄執行，例如：
    python P13-pose-features/features.py

這裡只放「全課要一致」的東西。模型與評估留在各節自己的程式裡。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PHOTOS = DATA / "photos"
MODELS = ROOT / "models"

# ---------------------------------------------------------------- MVTec AD

# 教師預先下載到本機（不隨 repo 散佈：CC BY-NC-SA 4.0）
MVTEC_ROOT = Path(os.environ.get("MVTEC_ROOT", Path.home() / "datasets/mvtec-ad"))
MVTEC_CATEGORIES = ["bottle", "screw", "hazelnut", "metal_nut", "carpet", "wood"]


def mvtec_paths(category: str):
    """回傳 dict：train（良品路徑 list）、test（路徑 list）、y（0 良品／1 瑕疵）、defect（類型）、mask（遮罩路徑或 None）。"""
    img = MVTEC_ROOT / "images"
    train = sorted(str(p) for p in (img / "train" / category / "good").glob("*.png"))
    test = sorted(str(p) for p in (img / "test" / category).glob("*/*.png"))
    defect = [Path(p).parent.name for p in test]
    y = np.array([0 if d == "good" else 1 for d in defect])
    mask = []
    for p, d in zip(test, defect):
        m = MVTEC_ROOT / "masks" / "test" / category / d / (Path(p).stem + "_mask.png")
        mask.append(str(m) if d != "good" and m.exists() else None)
    return dict(train=train, test=test, y=y, defect=defect, mask=mask)


def memory_calib_split(train_paths, frac=0.8, seed=0):
    """V08 決定：良品 80% 當「記憶」、20% 當「校正」（定門檻用，不碰測試集）。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_paths))
    cut = int(len(train_paths) * frac)
    return [train_paths[i] for i in order[:cut]], [train_paths[i] for i in order[cut:]]


# ---------------------------------------------------------------- 骨架（COCO 17 點）

KPTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
FPS = 10

# P14 決定：依「受試者」分組切割。S01–S18 訓練、S19–S24 測試（測試集只在 P16 最後打開一次）。
TEST_SUBJECTS = [f"S{i:02d}" for i in range(19, 25)]


def load_poses(path: Path | str = DATA / "pose/poses.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype={"activity": "str", "posture": "str"}, keep_default_na=False)


def subject_split(df: pd.DataFrame):
    is_test = df["subject_id"].isin(TEST_SUBJECTS)
    return df[~is_test].copy(), df[is_test].copy()


MIN_KPT_CONF = 0.3  # P13 決定：信心低於此的點視為「沒看到」


def keypoints(df: pd.DataFrame, mask_low=True) -> np.ndarray:
    """(n, 17, 3)：x, y, confidence。mask_low=True 時，低信心點的 x, y 改為 NaN。"""
    cols = [f"{k}_{a}" for k in KPTS for a in ("x", "y", "c")]
    k = df[cols].to_numpy(dtype=float, copy=True).reshape(len(df), 17, 3)
    if mask_low:
        k[k[..., 2] < MIN_KPT_CONF, :2] = np.nan
    return k


def _angle(a, b, c):
    """b 點的夾角（度），a-b-c。"""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# P13 決定的單格特徵（全部與畫面大小、人的遠近無關）
FRAME_FEATURES = ["torso_angle", "aspect", "hip_height", "head_hip_dy", "knee_angle", "hip_angle", "mean_conf"]


def frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """每一格（一列）→ 7 個姿勢特徵。單位：角度為度，長度以「軀幹長」為 1。"""
    k = keypoints(df)
    xy, c = k[..., :2], k[..., 2]
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        sh = np.nanmean(xy[:, [5, 6]], 1)
        hip = np.nanmean(xy[:, [11, 12]], 1)
        ank = np.nanmean(xy[:, [15, 16]], 1)
        knee = np.nanmean(xy[:, [13, 14]], 1)
        w = np.nanmax(xy[..., 0], 1) - np.nanmin(xy[..., 0], 1)
        h = np.nanmax(xy[..., 1], 1) - np.nanmin(xy[..., 1], 1)
    torso_len = np.linalg.norm(sh - hip, axis=1) + 1e-6
    d = sh - hip
    torso_angle = np.degrees(np.arctan2(np.abs(d[:, 0]), -d[:, 1]))  # 0＝直立、90＝水平
    out = pd.DataFrame(index=df.index)
    out["torso_angle"] = torso_angle
    out["aspect"] = h / (w + 1e-6)
    out["hip_height"] = (ank[:, 1] - hip[:, 1]) / torso_len
    out["head_hip_dy"] = (hip[:, 1] - xy[:, 0, 1]) / torso_len
    out["knee_angle"] = _angle(hip, knee, ank)
    out["hip_angle"] = _angle(sh, hip, knee)
    out["mean_conf"] = c.mean(1)  # 註：低信心點的 x, y 已是 NaN，特徵可能是 NaN（HGB 可直接處理）
    return out


# P15 決定：1 秒（10 格）滑動窗口的時間特徵
WINDOW = 10
WINDOW_FEATURES = FRAME_FEATURES + ["d_torso_angle", "d_hip_y", "max_hip_speed"]


def window_features(df: pd.DataFrame) -> pd.DataFrame:
    """每一格的特徵，加上「過去 1 秒」的變化量（同一個 seq_id 內計算，不跨序列）。"""
    orig_index = df.index
    assert orig_index.is_unique, "window_features 需要唯一的 index"
    df = df.sort_values(["seq_id", "frame"])
    f = frame_features(df)
    k = keypoints(df)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        hip = np.nanmean(k[:, [11, 12], :2], 1)
    f["hip_y_norm"] = hip[:, 1] / 480  # 以畫面高度為 1（同一支攝影機內可比）
    f["seq_id"] = df["seq_id"].to_numpy()
    g = f.groupby("seq_id", sort=False)
    f["d_torso_angle"] = f["torso_angle"] - g["torso_angle"].shift(WINDOW - 1)
    f["d_hip_y"] = f["hip_y_norm"] - g["hip_y_norm"].shift(WINDOW - 1)
    speed = g["hip_y_norm"].diff().abs()
    f["max_hip_speed"] = speed.groupby(f["seq_id"], sort=False).transform(
        lambda s: s.rolling(WINDOW, min_periods=1).max())
    return f[WINDOW_FEATURES].reindex(orig_index)  # 依「輸入順序」回傳


# ---------------------------------------------------------------- 預訓練特徵（V08 起共用，有磁碟快取）

CACHE = Path(os.environ.get("NTUNHS_CACHE", ROOT / "cache"))


def resnet_patch_features(paths, size=224, batch=16):
    """ResNet18（ImageNet）layer2＋layer3 → 每張圖 28×28 個小格、每格 384 維（V09 講解）。

    回傳 float32 array (n, 28, 28, 384)。依「路徑清單＋size」做磁碟快取，第二次執行直接讀檔。
    """
    import hashlib

    key = hashlib.sha256(("\n".join(map(str, paths)) + f"|{size}").encode()).hexdigest()[:16]
    f = CACHE / f"r18-l23-{key}.npy"
    if f.exists():
        return np.load(f).astype(np.float32)
    import torch
    import torchvision
    from PIL import Image

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights).eval()
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    hooked = {}
    model.layer2.register_forward_hook(lambda m, i, o: hooked.__setitem__("l2", o))
    model.layer3.register_forward_hook(lambda m, i, o: hooked.__setitem__("l3", o))
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            model(x)
            l2 = hooked["l2"]
            l3 = torch.nn.functional.interpolate(hooked["l3"], size=l2.shape[-2:], mode="bilinear")
            feat = torch.nn.functional.avg_pool2d(torch.cat([l2, l3], 1), 3, 1, 1)
            out.append(feat.permute(0, 2, 3, 1).numpy().astype(np.float16))
    arr = np.concatenate(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, arr)
    return arr.astype(np.float32)
