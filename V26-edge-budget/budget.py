"""V26：解析度 × 準確率 × 延遲的取捨表（本機 CPU；Pi 5 未實測）。

ResNet18 小格特徵（ONNX，onnxruntime）在 160／224／320 三種輸入解析度下：
- 異常偵測 AUROC：MVTec bottle、screw，小格 kNN（記憶庫抽 10%），train 前 60 張（80% 記憶／20% 校正）、test 全部
- 每張圖延遲 p50：特徵抽取（onnxruntime 4 執行緒）＋ kNN 打分數（torch 4 執行緒），batch 1

特徵與分數存快取（cache/onnx/feat-*.npy、scores-*.npz）。延遲因機器而異（講義不貼毫秒）。

    python V26-edge-budget/budget.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "V24-onnx-export")
from edge_lib import (R18_ONNX, auroc, export_r18, ort_patches, ort_session, p50_ms, preprocess,  # noqa: E402
                      subset, subset_scores, sys_path_v10)

sys_path_v10()
from anomaly_lib import patch_maps, scores_patch  # noqa: E402

SIZES = [160, 224, 320]
CATS = ["bottle", "screw"]
export_r18()

rows = []
for size in SIZES:
    row = {"解析度": size, "小格": f"{size // 8}×{size // 8}"}
    for cat in CATS:
        s = subset_scores(cat, size)
        y, th = s["y"], s["calib"].max()  # V08 的預設門檻：校正良品分數最大值
        row[f"{cat} AUROC"] = round(auroc(y, s["test"]), 3)
        row[f"{cat} recall"] = round(float((s["test"][y == 1] > th).mean()), 3)
        row[f"{cat} 誤報"] = int((s["test"][y == 0] > th).sum())
    row["記憶庫小格"] = int(s["bank_n"])
    rows.append(row)

print("== 1. 解析度 vs 異常偵測（小格 kNN、記憶庫 10%、train 前 60 張；門檻＝校正良品最大值）==")
t = pd.DataFrame(rows)
print(t.to_string(index=False))

print("\n== 1b. 對照：224、train 全部（V10 的設定，common 快取）==")
from common import mvtec_paths  # noqa: E402

for cat in CATS:
    P = mvtec_paths(cat)
    s = scores_patch(cat)
    print(f"{cat}：train {len(P['train'])} 張，AUROC {auroc(P['y'], s['test']):.3f}")

print("\n== 2. 計算量：相對 224 的像素比（≈特徵抽取的計算量）與 kNN 距離計算量 ==")
for r in rows:
    size, n = r["解析度"], (r["解析度"] // 8) ** 2
    print(f"{size}：像素比 {size ** 2 / 224 ** 2:.2f}，每張 {n} 個小格 × 記憶庫 {r['記憶庫小格']} = "
          f"{n * r['記憶庫小格'] / 1e6:.1f} 百萬次距離")

print("\n== 3. 本機延遲 p50（毫秒／張，batch 1，4 執行緒；因機器而異，講義不貼）==")
sess = ort_session(R18_ONNX, threads=4)
one = subset("bottle")[1][:1]
rng = np.random.default_rng(0)
lat = []
for r in rows:
    size = r["解析度"]
    x = preprocess(one, size)
    feat_ms = p50_ms(lambda: ort_patches(sess, x), n=10, warmup=2)
    f = ort_patches(sess, x)
    bank = rng.standard_normal((r["記憶庫小格"], 384)).astype(np.float32)  # 只量速度：大小對就好
    knn_ms = p50_ms(lambda: patch_maps(f, bank), n=10, warmup=2)
    lat.append({"解析度": size, "特徵": round(feat_ms), "kNN": round(knn_ms), "合計": round(feat_ms + knn_ms)})
print(pd.DataFrame(lat).to_string(index=False))
