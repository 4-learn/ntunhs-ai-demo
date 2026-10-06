"""V28｜製造組 MVP 的模型端：小格 kNN 記憶庫（定版成檔案）、門檻規則、推論器、輸送帶取像。

V33–V35 也 import 這支：

    import sys; from common import ROOT
    sys.path.insert(0, str(ROOT / "V28-mvp-defect"))
    from defect_model import build_model, load_model, PatchKnnDetector, ConveyorSource, conveyor_order

設計（V28 講解）：
- 特徵一律走 common.resnet_patch_features 的快取（傳完整 P["train"]／P["test"]），不另外抽。
- 記憶庫＝記憶良品的小格抽 10%（rng 0），與 V10 相同；存成 cache/models/patch-knn-{類別}.npy
  （float16，可重現、可算 hash）。記憶庫是 MVTec 影像的衍生物（CC BY-NC-SA 4.0），所以放在 cache/
  （.gitignore），**不進公開 repo**；各機器用 build_model() 自己產生。
- 分數：原始分數＝最不像的那一小格到最近正常小格的距離（V10）；事件 API 只收 0–1，
  所以除以固定尺度 scale（＝校正良品最大值 × 2，存在 .json）再截到 1。換算是單調的，判斷結果不變。
- 門檻：直接呼叫 V12 的 anomaly_lib.cost_threshold／valid_test_split（校正良品＋每類 10 張驗證瑕疵，seed 0，
  成本 20×漏報＋1×誤報，同成本取較高門檻）。管線用「score ≥ 門檻」，所以門檻往上取 3 位小數。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import common
from common import CACHE, ROOT, mvtec_paths, resnet_patch_features

for _d in ("V08-anomaly-intro", "V27-event-api", "V31-mock-tests"):
    sys.path.insert(0, str(ROOT / _d))
from anomaly_lib import BANK_FRAC, build_bank, cost_threshold, split_index, valid_test_split  # noqa: E402

COST_MISS, COST_FA = 20, 1          # 王課長的數字（V12）
VERSION = "2026.01"
THREADS = int(os.environ.get("TORCH_THREADS", "2"))


def valid_index(y, seed=0):
    """V12：從測試集的瑕疵裡固定抽 10 張當「驗證瑕疵」（定門檻用），其餘留給最終測試。"""
    return valid_test_split(y, seed=seed)[0]


def nn_max(feat, bank_t, bb):
    """一張圖 (28, 28, 384) → (28, 28) 小格距離圖。bank_t 是 torch tensor（記憶庫）。"""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(feat.reshape(-1, feat.shape[-1]), dtype=np.float32))
    d2 = torch.addmm(bb, x, bank_t.T, alpha=-2).min(1).values + (x * x).sum(1)
    return d2.clamp_min(0).sqrt().numpy().reshape(feat.shape[:2])


MODEL_DIR = CACHE / "models"


def model_paths(category):
    return MODEL_DIR / f"patch-knn-{category}.npy", MODEL_DIR / f"patch-knn-{category}.json"


def build_model(category, rule="cost"):
    """從 MVTec 良品建記憶庫、算校正分數、用 V12 規則定門檻；寫出 .npy（記憶庫）與 .json（模型卡）。

    rule="cost"（V12 成本規則）或 "max"（V08 校正良品最大值）。檔案內容可重現：同樣輸入 → 同樣 sha256。
    """
    import torch

    torch.set_num_threads(THREADS)
    P = mvtec_paths(category)
    mem, cal = split_index(len(P["train"]))
    tr = resnet_patch_features(P["train"])                 # 完整清單 → 讀快取
    bank = build_bank(tr, mem, BANK_FRAC).astype(np.float16)  # 快取本來就是 float16，轉回去不失真
    calf = tr[cal]
    del tr
    bt = torch.from_numpy(bank.astype(np.float32))
    bb = (bt * bt).sum(1)
    calib = np.array([nn_max(f, bt, bb).max() for f in calf])
    vi = valid_index(P["y"])
    te = resnet_patch_features(P["test"])
    valid = np.array([nn_max(te[i], bt, bb).max() for i in vi])
    del te
    scale = round(float(calib.max()) * 2, 3)
    # V12 的函式：分數 > 門檻才剔除；同成本取較高門檻
    t_raw = cost_threshold(calib, valid)[0] if rule == "cost" else float(calib.max())
    miss, fa = int((valid <= t_raw).sum()), int((calib > t_raw).sum())
    cost = COST_MISS * miss + COST_FA * fa
    thr = np.ceil(t_raw / scale * 1000) / 1000               # V12 是「> 門檻」，管線是「≥」：往上取 3 位（講解第 3 小節）
    npy, card = model_paths(category)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy, bank)
    info = {"name": f"patch-knn-{category}", "version": VERSION, "category": category,
            "bank_frac": BANK_FRAC, "bank_shape": list(bank.shape), "scale": scale,
            "n_memory": len(mem), "n_calib": len(cal), "valid_index": vi.tolist(), "rule": rule,
            "threshold_raw": round(t_raw, 4), "threshold": float(thr),
            "valid_cost": cost, "valid_miss": miss, "valid_fa": fa,
            "calib_max_raw": round(float(calib.max()), 4)}
    card.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def load_model(category, rebuild=False):
    """讀定版模型；檔案不在就建一次（結果可重現）。回傳 (記憶庫, 模型卡 dict)。"""
    npy, card = model_paths(category)
    if rebuild or not (npy.exists() and card.exists()):
        build_model(category)
    return np.load(npy), json.loads(card.read_text(encoding="utf-8"))


def image_key(image):
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()


class CachedFeatures:
    """依「影像內容」查 common 快取裡的特徵：輸送帶上的照片就是 MVTec 測試照片，算過就不再算。

    影像不在表裡會丟 KeyError——不會默默拿錯一張的特徵。
    """

    def __init__(self, category):
        P = mvtec_paths(category)
        self.feats = resnet_patch_features(P["test"])
        self.index = {image_key(cv2.imread(p)): i for i, p in enumerate(P["test"])}

    def __call__(self, image):
        return self.feats[self.index[image_key(image)]]


def online_features(image, tmp=Path("/tmp/ntunhs/V28/online")):
    """真正的線上流程：把這一格存成 PNG（無損），交給同一支 common.resnet_patch_features 抽特徵。

    檔名用內容 hash（快取以路徑為鍵，同名不同圖會拿到舊特徵）；暫時把 common.CACHE 指到 /tmp，
    不污染教師的快取資料夾。
    """
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / f"{image_key(image)}.png"
    cv2.imwrite(str(p), image)
    old = common.CACHE
    common.CACHE = tmp / "cache"
    try:
        return resnet_patch_features([str(p)])[0]
    finally:
        common.CACHE = old


class PatchKnnDetector:
    """V31 的推論器介面：predict(image) → {"score": 0–1, "evidence": {...}}；另外記下原始分數供評估。"""

    def __init__(self, bank, card, featurize, part_ids=None):
        import torch

        torch.set_num_threads(THREADS)
        self.card, self.featurize = card, featurize
        self.bank = torch.from_numpy(bank.astype(np.float32))
        self.bb = (self.bank * self.bank).sum(1)
        self.part_ids = iter(part_ids) if part_ids is not None else None
        self.n, self.last = 0, None

    def predict(self, image):
        assert image.ndim == 3 and image.dtype == np.uint8, "推論器只收 HxWx3 uint8"
        dmap = nn_max(self.featurize(image), self.bank, self.bb)
        raw = float(dmap.max())
        r, c = np.unravel_index(int(dmap.argmax()), dmap.shape)
        h, w = image.shape[:2]
        self.n += 1
        pid = next(self.part_ids) if self.part_ids is not None else f"P{self.n:04d}"
        ev = {"part_id": pid, "category": self.card["category"],
              "heatmap_max_xy": [int((c + 0.5) * w / dmap.shape[1]), int((r + 0.5) * h / dmap.shape[0])]}
        self.last = {"part_id": pid, "raw": raw, "score": min(raw / self.card["scale"], 1.0)}
        return {"score": self.last["score"], "evidence": ev}


@dataclass
class Shot:
    path: str
    truth: int
    defect: str


def conveyor_order(category, seed=0):
    """輸送帶依序拍到的工件：測試集扣掉 10 張驗證瑕疵（定門檻用過的不能再拿來驗收），亂序（rng seed）。"""
    P = mvtec_paths(category)
    vi = set(valid_index(P["y"]).tolist())
    keep = [i for i in range(len(P["test"])) if i not in vi]
    order = np.random.default_rng(seed).permutation(keep)
    return [Shot(P["test"][i], int(P["y"][i]), P["defect"][i]) for i in order]


class ConveyorSource:
    """V31 的取像介面：grab() → Frame。依序讀「輸送帶上的工件照片」，時間由 clock 決定。"""

    def __init__(self, shots, clock):
        from pipeline import Frame  # V31

        self.Frame, self.shots, self.clock, self.i = Frame, shots, clock, 0

    def grab(self):
        s = self.shots[self.i]
        self.i += 1
        img = cv2.imread(s.path)              # 灰階 PNG 也讀成 3 通道 BGR
        return self.Frame(img, self.clock())

    def __len__(self):
        return len(self.shots)


def evaluate(truth, alarm):
    """truth、alarm 都是 0/1 序列 → dict(漏報, 誤報, recall, precision)。"""
    truth, alarm = np.asarray(truth), np.asarray(alarm)
    tp = int(((truth == 1) & (alarm == 1)).sum())
    miss = int(((truth == 1) & (alarm == 0)).sum())
    fa = int(((truth == 0) & (alarm == 1)).sum())
    return {"漏報": miss, "誤報": fa, "recall": tp / max(tp + miss, 1), "precision": tp / max(tp + fa, 1)}



def run_line(category, client, model_info, prefix="line1cam1", site="精工五金", start=None, step_s=3):
    """一條線跑一輪（V33、V34 用）：輸送帶 → 推論器 → V31 Pipeline → client（V27）。

    model_info：{"name", "version", "sha256_12"}（寫進每筆事件）。回傳 (每顆工件一列的 DataFrame, 模型卡)。
    """
    from datetime import datetime

    import pandas as pd
    from pipeline import TW, ApiSender, PipeConfig, Pipeline, StepClock

    bank, card = load_model(category)
    shots = conveyor_order(category)
    det = PatchKnnDetector(bank, card, CachedFeatures(category), part_ids=[f"P{i + 1:05d}" for i in range(len(shots))])
    start = start or datetime(2026, 1, 12, 8, 0, tzinfo=TW)
    pipe = Pipeline(ConveyorSource(shots, StepClock(start, step_s)), det, ApiSender(client),
                    PipeConfig(source="defect", site=site, camera_id=prefix, prefix=prefix,
                               threshold=card["threshold"], model=model_info, cooldown_s=0))
    rows = []
    for s in shots:
        st = pipe.step()
        rows.append({"part_id": det.last["part_id"], "truth": s.truth, "defect": s.defect,
                     "raw": det.last["raw"], "score": det.last["score"], "status": st})
    return pd.DataFrame(rows), card
