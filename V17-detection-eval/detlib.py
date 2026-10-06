"""V17｜瑕疵偵測評估的小工具：讀 YOLO 標註、IoU、逐框配對、影像層級判斷。V18 也 import 這支。

偵測結果會快取到 cache/（以權重檔 sha256 與 split 為鍵），第二次執行直接讀檔。
"""
from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("YOLO_OFFLINE", "1")

from pathlib import Path

import numpy as np

from common import CACHE, MODELS

DS = CACHE / "yolo-defects"
WEIGHTS = Path(os.environ.get("DEFECT_WEIGHTS", MODELS / "defect-yolo11n.pt"))  # 可用環境變數換權重比較
SIZE = 320
FN_COST, FP_COST = 20, 1  # 王課長：漏到客戶一次＝多一次人工複檢的 20 倍
GRID = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # 0.01＝detect() 的下限：「有任何框」


def iou(a, b) -> float:
    """兩個 xyxy 框的 IoU＝交集面積 / 聯集面積（ML 課 19 節）。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def names(split: str) -> list[str]:
    return sorted(p.stem for p in (DS / "images" / split).glob("*.png"))


def load_labels(split: str, name: str) -> list[list[float]]:
    """YOLO 標註（類別 中心x 中心y 寬 高，0–1）→ 320×320 上的像素 xyxy。空檔＝良品＝[]。"""
    out = []
    for line in (DS / "labels" / split / f"{name}.txt").read_text().splitlines():
        if line.strip():
            _, x, y, w, h = map(float, line.split())
            out.append([(x - w / 2) * SIZE, (y - h / 2) * SIZE, (x + w / 2) * SIZE, (y + h / 2) * SIZE])
    return out


def info(name: str) -> tuple[str, str]:
    """檔名 metal_nut-bent-001 → ('metal_nut', 'bent')。"""
    cat, rest = name.split("-", 1)
    return cat, rest.rsplit("-", 1)[0]


def weights_sha(path=None) -> str:
    path = path or WEIGHTS
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def detect(split: str, min_conf: float = 0.01) -> dict[str, list[tuple[float, list[float]]]]:
    """對 split 的每張圖推論，回傳 {圖名: [(conf, xyxy), ...]}。門檻先放很低，之後再掃。"""
    f = CACHE / f"det-{weights_sha()}-{split}-{min_conf}.json"
    if f.exists():
        return json.loads(f.read_text())
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    model = YOLO(str(WEIGHTS))
    out = {}
    paths = [str(DS / "images" / split / f"{n}.png") for n in names(split)]
    for n, r in zip(names(split), model.predict(paths, conf=min_conf, imgsz=SIZE, batch=8, verbose=False,
                                                 stream=True, device="cpu")):
        out[n] = [(round(c, 4), [round(v, 1) for v in b]) for c, b in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    f.write_text(json.dumps(out))
    return out


def match(dets, gts, threshold: float, iou_min: float = 0.5):
    """單張圖：信心由高到低，每個預測框找 IoU 最大且還沒被配過的標註框。回傳 TP, FP, FN。"""
    preds = sorted([d for d in dets if d[0] >= threshold], key=lambda d: -d[0])
    used, tp = set(), 0
    for _, box in preds:
        best, best_j = 0.0, None
        for j, t in enumerate(gts):
            if j not in used and iou(box, t) > best:
                best, best_j = iou(box, t), j
        if best >= iou_min:
            tp += 1
            used.add(best_j)
    return tp, len(preds) - tp, len(gts) - tp


def box_sweep(dets, split, thresholds=GRID, iou_min=0.5):
    """框層級：每個門檻的 TP／FP／FN、precision、recall。"""
    rows = []
    for t in thresholds:
        tp = fp = fn = 0
        for n in names(split):
            a, b, c = match(dets[n], load_labels(split, n), t, iou_min)
            tp, fp, fn = tp + a, fp + b, fn + c
        rows.append({"門檻": t, "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
                     "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0})
    return rows


def image_scores(dets, split) -> tuple[np.ndarray, np.ndarray]:
    """影像層級：分數＝這張圖最高的框信心（沒有框＝0）；y＝這張圖有沒有標註框。"""
    ns = names(split)
    s = np.array([max([c for c, _ in dets[n]], default=0.0) for n in ns])
    y = np.array([int(len(load_labels(split, n)) > 0) for n in ns])
    return s, y


def image_metrics(y, score, thr) -> dict:
    y, pred = np.asarray(y), np.asarray(score) >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
            "成本": FN_COST * fn + FP_COST * fp}


def pick_threshold(y, score, grid=GRID) -> float:
    """成本 20×漏報＋1×誤報 最低者；同分取較大的門檻（少吵一點）。"""
    costs = [image_metrics(y, score, t)["成本"] for t in grid]
    return max(t for t, c in zip(grid, costs) if c == min(costs))


def per_10k(y, pred, rate=0.001, n=10_000) -> dict:
    """把 test 的漏報率、良品誤剔率換算成產線：每 n 件、瑕疵率 rate（V12 第 5 小節）。"""
    y, pred = np.asarray(y), np.asarray(pred).astype(bool)
    n_bad = n * rate
    fpr, fnr = pred[y == 0].mean(), (~pred[y == 1]).mean()
    return {"剔除良品": round(fpr * (n - n_bad)), "漏報": round(fnr * n_bad, 1),
            "成本": round(FN_COST * fnr * n_bad + FP_COST * fpr * (n - n_bad))}


def pick_threshold_line(y, score, grid=GRID, rate=0.001) -> float:
    """依產線瑕疵率加權：每 10,000 件的期望成本最低者（同分取較大門檻）。"""
    costs = [per_10k(y, np.asarray(score) >= t, rate)["成本"] for t in grid]
    return max(t for t, c in zip(grid, costs) if c == min(costs))
