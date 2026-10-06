"""V3 段（V08–V13）共用的異常分數；V13、V18、V28 也直接 import 這支。

從 demo/ 執行的腳本這樣用：
    import sys; from common import ROOT
    sys.path.insert(0, str(ROOT / "V08-anomaly-intro"))
    from anomaly_lib import scores_patch

三種方法與 V08–V10 講義裡的程式一致：
- A 像素差：記憶良品平均成一張灰階 128×128 的「平均良品」，分數＝與它的平均絕對差（V08）
- B 整張特徵 kNN：28×28 小格特徵平均成一個 384 維向量，分數＝到最近記憶良品的距離（V09）
- C 小格 kNN（PatchCore 簡化版）：記憶良品的所有小格抽 10%（rng 0）當記憶庫，
  每格找最近的正常小格，整張分數＝最不像的那一格（V10）

特徵一律用 common.resnet_patch_features，傳完整 train／test 清單（快取鍵），再用索引切記憶／校正。
分數也存到 common.CACHE（scores-*.npz），第二次執行直接讀檔。
"""
from __future__ import annotations

import hashlib

import numpy as np

from common import CACHE, memory_calib_split, mvtec_paths, resnet_patch_features

BANK_FRAC = 0.1


def split_index(n):
    """memory_calib_split 作用在索引上：回傳 (記憶索引, 校正索引)。"""
    mem, cal = memory_calib_split(list(range(n)))
    return np.array(mem), np.array(cal)


def nn_dist(query, bank, chunk=None):
    """query (m, d) 每一列到 bank (n, d) 最近一列的歐氏距離。分批算，記憶體不爆。"""
    import torch

    torch.set_num_threads(4)
    b = torch.from_numpy(np.ascontiguousarray(bank, dtype=np.float32))
    bb = (b * b).sum(1)
    chunk = chunk or max(64, int(5e7 // len(b)))  # 每批距離矩陣約 200 MB
    out = np.empty(len(query), np.float32)
    for i in range(0, len(query), chunk):
        x = torch.from_numpy(np.ascontiguousarray(query[i:i + chunk], dtype=np.float32))
        d2 = torch.addmm(bb, x, b.T, alpha=-2).min(1).values + (x * x).sum(1)
        out[i:i + chunk] = d2.clamp_min(0).sqrt().numpy()
    return out


def gray(path, size=128):
    from PIL import Image

    return np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.float32) / 255


def _cached(name, cat, paths, fn):
    key = hashlib.sha256("\n".join(paths).encode()).hexdigest()[:12]
    f = CACHE / f"scores-{name}-{cat}-{key}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    out = fn()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, **out)
    return out


def scores_pixel(cat):
    P = mvtec_paths(cat)
    mem, cal = split_index(len(P["train"]))

    def run():
        mean_img = np.mean([gray(P["train"][i]) for i in mem], axis=0)
        s = lambda ps: np.array([np.abs(gray(p) - mean_img).mean() for p in ps])
        return dict(calib=s([P["train"][i] for i in cal]), test=s(P["test"]))

    return _cached("A", cat, P["train"] + P["test"], run)


def scores_global(cat):
    P = mvtec_paths(cat)
    mem, cal = split_index(len(P["train"]))

    def run():
        tr = resnet_patch_features(P["train"]).mean(axis=(1, 2))
        te = resnet_patch_features(P["test"]).mean(axis=(1, 2))
        return dict(calib=nn_dist(tr[cal], tr[mem]), test=nn_dist(te, tr[mem]))

    return _cached("B", cat, P["train"] + P["test"], run)


def build_bank(train_feats, mem, frac=BANK_FRAC, seed=0):
    bank = train_feats[mem].reshape(-1, train_feats.shape[-1])
    if frac < 1:
        rng = np.random.default_rng(seed)
        bank = bank[rng.choice(len(bank), int(len(bank) * frac), replace=False)]
    return bank


def patch_maps(feats, bank):
    """(n, 28, 28, 384) → (n, 28, 28)：每一小格到最近正常小格的距離。"""
    n, h, w, d = feats.shape
    return nn_dist(feats.reshape(-1, d), bank).reshape(n, h, w)


def scores_patch(cat, frac=BANK_FRAC):
    """回傳 calib、test（每張分數）與 test_map（每張 28×28 小格分數，V11 熱圖用）。"""
    P = mvtec_paths(cat)
    mem, cal = split_index(len(P["train"]))

    def run():
        tr = resnet_patch_features(P["train"])
        bank = build_bank(tr, mem, frac)
        cal_map = patch_maps(tr[cal], bank)
        del tr
        test_map = patch_maps(resnet_patch_features(P["test"]), bank)
        return dict(calib=cal_map.reshape(len(cal), -1).max(1),
                    test=test_map.reshape(len(test_map), -1).max(1), test_map=test_map)

    return _cached(f"C{frac:g}", cat, P["train"] + P["test"], run)


METHODS = {"A 像素差": scores_pixel, "B 整張kNN": scores_global, "C 小格kNN": scores_patch}


# ---------------------------------------------------------------- V12 定版：少量瑕疵驗證＋成本門檻

N_VALID_DEFECTS = 10
COST_FN, COST_FP = 20, 1      # 王課長：漏到客戶一次＝多一次人工複檢的 20 倍


def valid_test_split(y, n=N_VALID_DEFECTS, seed=0):
    """test 的瑕疵固定抽 n 張（rng seed 0）當驗證；回傳 (驗證瑕疵索引, 最終測試索引)。

    最終測試＝全部 test 良品＋其餘瑕疵。驗證的良品用校正良品（train 的 20%）。
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    val = np.sort(rng.choice(np.where(y == 1)[0], n, replace=False))
    final = np.setdiff1d(np.arange(len(y)), val)
    return val, final


def cost_threshold(good_scores, defect_scores, c_fn=COST_FN, c_fp=COST_FP):
    """在驗證集（校正良品＋10 張瑕疵）上掃門檻，回傳 (門檻, 成本)。分數 > 門檻才剔除。

    候選門檻＝驗證分數本身（以及比最小分數再小一點）；同成本時取較高的門檻（少剔一點良品）。
    """
    s = np.concatenate([good_scores, defect_scores])
    cands = np.unique(np.concatenate([s, [s.min() - 1e-6]]))
    best = None
    for t in cands[::-1]:
        cost = c_fn * int((defect_scores <= t).sum()) + c_fp * int((good_scores > t).sum())
        if best is None or cost < best[1]:
            best = (float(t), cost)
    return best


def confusion(y, scores, thr):
    """分數 > 門檻＝剔除。回傳 dict：recall、precision、漏報、誤報、成本。"""
    y = np.asarray(y)
    pred = scores > thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return dict(recall=tp / (tp + fn) if tp + fn else 0.0, precision=tp / (tp + fp) if tp + fp else 0.0,
                漏報=fn, 誤報=fp, 成本=COST_FN * fn + COST_FP * fp)
