"""V18｜共用：把 V15 的 val／test 整理成一張表，每張圖同時有 YOLO 分數與 V10 小格 kNN 分數。

小格 kNN 直接用 V08 的 anomaly_lib.scores_patch（記憶庫＝train 記憶良品、小格抽 10%，分數有快取），
門檻規則用 anomaly_lib.cost_threshold（20×漏報＋1×誤報；同成本取較高門檻；分數 > 門檻才剔除）。
"""
import sys

import numpy as np
import pandas as pd

from common import ROOT

sys.path.insert(0, str(ROOT / "V08-anomaly-intro"))
sys.path.insert(0, str(ROOT / "V17-detection-eval"))
from anomaly_lib import BANK_FRAC, cost_threshold, scores_patch  # noqa: E402
from detlib import detect, image_scores, info, names  # noqa: E402

CATS = ["metal_nut", "screw"]


def index_of(name):
    """V15 檔名最後的數字＝MVTec test 清單（mvtec_paths(cat)["test"]）的索引。"""
    return int(name.rsplit("-", 1)[1])


def table(frac=BANK_FRAC):
    """V15 的 val＋test 一張圖一列：cat、defect、y、knn、yolo。另外回傳 kNN 熱圖、偵測結果、校正良品分數。"""
    rows = []
    for split in ("val", "test"):
        for n in names(split):
            cat, d = info(n)
            rows.append(dict(split=split, name=n, cat=cat, defect=d, y=int(d != "good")))
    df = pd.DataFrame(rows)
    S = {cat: scores_patch(cat, frac) for cat in CATS}
    df["knn"] = [float(S[c]["test"][index_of(n)]) for c, n in zip(df.cat, df.name)]
    det = {s: detect(s) for s in ("val", "test")}
    yolo = {}
    for s in ("val", "test"):
        sc, _ = image_scores(det[s], s)
        yolo.update(dict(zip(names(s), sc)))
    df["yolo"] = df.name.map(yolo)
    return df, {c: S[c]["test_map"] for c in CATS}, det, {c: S[c]["calib"] for c in CATS}


def line_threshold(good, bad, rate=0.001, n=10_000):
    """產線加權（V12 Workshop 第 3 題）：每 n 件、瑕疵率 rate 的期望成本最低；候選＝分數本身，同成本取較高門檻。"""
    good, bad = np.asarray(good), np.asarray(bad)
    s = np.concatenate([good, bad])
    best = None
    for t in np.unique(np.concatenate([s, [s.min() - 1e-6]]))[::-1]:
        cost = 20 * n * rate * (bad <= t).mean() + n * (1 - rate) * (good > t).mean()
        if best is None or cost < best[1]:
            best = (float(t), cost)
    return best


def thresholds(val_df, col, per_cat=True, rule=cost_threshold):
    """依 V15 val 選門檻。per_cat：每個產品各自一個門檻（kNN 每個產品各有記憶庫）；否則兩個產品共用（YOLO）。"""
    groups = CATS if per_cat else [None]
    out = {}
    for c in groups:
        v = val_df if c is None else val_df[val_df.cat == c]
        t = rule(v[v.y == 0][col].to_numpy(), v[v.y == 1][col].to_numpy())[0]
        out.update({k: t for k in CATS} if c is None else {c: t})
    return out


def predict(df, col, thr):
    """分數 > 門檻＝剔除（與 anomaly_lib.confusion 一致）。"""
    return np.array([int(s > thr[c]) for s, c in zip(df[col], df.cat)])


def knn_thresholds(val_df):
    return thresholds(val_df, "knn")


def knn_pred(df, thr):
    return predict(df, "knn", thr)
