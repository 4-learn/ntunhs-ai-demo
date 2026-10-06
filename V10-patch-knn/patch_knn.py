"""V10｜切成小格比：PatchCore 簡化版——小格記憶庫、最不像的小格＝分數、記憶庫大小的取捨。

第一次執行要算 1%／10%／100% 三種記憶庫的距離（screw 100% 在 4 執行緒 CPU 約 10 分鐘），
結果存到 demo/cache/scores-C*.npz；之後直接讀檔。
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ROOT, memory_calib_split, mvtec_paths, resnet_patch_features

sys.path.insert(0, str(ROOT / "V08-anomaly-intro"))
from anomaly_lib import build_bank, nn_dist, scores_global, scores_patch  # noqa: E402

P = mvtec_paths("bottle")
mem, cal = memory_calib_split(list(range(len(P["train"]))))
tr = resnet_patch_features(P["train"])
print("== 1. 小格記憶庫：bottle ==")
all_patches = tr[mem].reshape(-1, 384)
print(f"記憶良品 {len(mem)} 張 × 784 格 = {len(all_patches)} 個小格")
rng = np.random.default_rng(0)
bank = all_patches[rng.choice(len(all_patches), len(all_patches) // 10, replace=False)]
print(f"固定 rng(0) 抽 10%：{len(bank)} 個小格，{bank.nbytes / 1e6:.1f} MB")

print("\n== 2. 一張瑕疵圖的 784 個小格分數 ==")
te = resnet_patch_features(P["test"])
i = P["defect"].index("broken_small")
d = nn_dist(te[i].reshape(-1, 384), bank).reshape(28, 28)
print(f"{P['defect'][i]} 第 {i} 張：小格分數 shape {d.shape}")
print(f"中位數 {np.median(d):.3f}、平均 {d.mean():.3f}、最大 {d.max():.3f}（在第 {tuple(int(v) for v in np.unravel_index(d.argmax(), d.shape))} 格，列、行）")
j = P["defect"].index("good")
g = nn_dist(te[j].reshape(-1, 384), bank).reshape(28, 28)
print(f"對照：測試良品第 {j} 張 中位數 {np.median(g):.3f}、平均 {g.mean():.3f}、最大 {g.max():.3f}")


def evaluate(cat, frac=0.1):
    y = mvtec_paths(cat)["y"]
    s = scores_patch(cat, frac)
    pred = s["test"] > s["calib"].max()
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    return dict(AUROC=roc_auc_score(y, s["test"]), recall=tp / y.sum(),
                precision=tp / (tp + fp) if tp + fp else 0.0, 漏報=int(y.sum()) - tp, 誤報=fp)


print("\n== 3. 方法 C（記憶庫 10%）vs 方法 B，門檻＝校正良品最大值 ==")
rows = []
for cat in ["bottle", "screw"]:
    y = mvtec_paths(cat)["y"]
    rows.append({"類別": cat, "B AUROC": roc_auc_score(y, scores_global(cat)["test"]), **evaluate(cat)})
print(pd.DataFrame(rows).set_index("類別").to_string(float_format="%.3f"))

print("\n== 4. 為什麼取最大值，不取平均 ==")
for cat in ["bottle", "screw"]:
    y = mvtec_paths(cat)["y"]
    m = scores_patch(cat)["test_map"].reshape(len(y), -1)
    print(f"{cat}：整張分數＝最大值 AUROC {roc_auc_score(y, m.max(1)):.3f}；"
          f"＝平均 {roc_auc_score(y, m.mean(1)):.3f}；＝前 1% 小格平均 {roc_auc_score(y, np.sort(m, 1)[:, -8:].mean(1)):.3f}")

print("\n== 5. 記憶庫大小的取捨：1%、10%、100% ==")
rows = []
timing = []
for cat in ["bottle", "screw"]:
    P = mvtec_paths(cat)
    mem, _ = memory_calib_split(list(range(len(P["train"]))))
    tr = resnet_patch_features(P["train"])
    for frac in (0.01, 0.1, 1.0):
        n = int(len(mem) * 784 * frac) if frac < 1 else len(mem) * 784
        r = evaluate(cat, frac)
        rows.append({"類別": cat, "記憶庫": f"{frac:.0%}", "小格數": n, "MB": n * 384 * 4 / 1e6,
                     "每張距離計算(百萬次)": 784 * n / 1e6, "AUROC": r["AUROC"], "recall": r["recall"]})
        bank = build_bank(tr, mem, frac)
        one = tr[mem[0]].reshape(-1, 384)
        t0 = time.perf_counter()
        nn_dist(one, bank)
        timing.append(f"{cat} {frac:.0%}：{time.perf_counter() - t0:.2f} 秒／張")
    del tr
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))
print("\n（速度參考，因機器與負載而異，講義不貼）", "；".join(timing))
