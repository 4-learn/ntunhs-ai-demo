"""V10｜讓 AI 寫，然後挑錯：抽圖而不是抽小格、抽樣不固定種子、一次算完整距離矩陣。"""
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from common import ROOT, memory_calib_split, mvtec_paths, resnet_patch_features

sys.path.insert(0, str(ROOT / "V08-anomaly-intro"))
from anomaly_lib import nn_dist  # noqa: E402

P = mvtec_paths("screw")
y = P["y"]
tr = resnet_patch_features(P["train"])
te = resnet_patch_features(P["test"])
mem, cal = memory_calib_split(list(range(len(P["train"]))))


def score(feats, bank):
    return nn_dist(feats.reshape(-1, 384), bank).reshape(len(feats), -1).max(1)


def sample(patches, frac, seed):
    rng = np.random.default_rng(seed)
    return patches[rng.choice(len(patches), int(len(patches) * frac), replace=False)]


print("== 錯誤 1：「抽 10%」抽成 10% 的圖，不是 10% 的小格 ==")
rng = np.random.default_rng(0)
some = rng.choice(mem, len(mem) // 10, replace=False)
for name, bank in [(f"AI 版（{len(some)} 張圖的全部小格）", tr[some].reshape(-1, 384)),
                   ("正確版（256 張圖的小格抽 10%）", sample(tr[mem].reshape(-1, 384), 0.1, 0))]:
    s_cal, s_test = score(tr[cal], bank), score(te, bank)
    pred = s_test > s_cal.max()
    print(f"{name}：{len(bank)} 個小格｜AUROC {roc_auc_score(y, s_test):.3f}｜"
          f"recall {(pred & (y == 1)).sum() / y.sum():.3f}")

print("\n== 錯誤 2：抽樣沒有固定種子（記憶庫 1%）==")
patches = tr[mem].reshape(-1, 384)
aucs = [roc_auc_score(y, score(te, sample(patches, 0.01, seed))) for seed in range(3)]
print("seed 0、1、2 的 AUROC：", "、".join(f"{a:.3f}" for a in aucs))
print(f"差距 {max(aucs) - min(aucs):.3f}：每次執行數字都不同，成績表就沒辦法對")

print("\n== 錯誤 3：torch.cdist(全部測試小格, 全部記憶小格) ==")
n_q, n_b = len(te) * 784, len(mem) * 784
print(f"測試小格 {n_q} × 記憶小格 {n_b} × 4 bytes = {n_q * n_b * 4 / 1e9:.0f} GB 的距離矩陣")
print("→ 要分批：每批只算一部分查詢，取完最小值就丟掉（anomaly_lib.nn_dist 的做法）")
