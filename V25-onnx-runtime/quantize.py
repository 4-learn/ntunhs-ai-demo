"""V25：int8 動態量化（onnxruntime.quantization.quantize_dynamic）——檔案變小，分數變多少、有沒有變快？

比較同一個 ResNet18 小格特徵抽取器的 fp32 與 int8 版本：
檔案大小、特徵差異、MVTec bottle／screw 的異常偵測 AUROC（小格 kNN，記憶庫抽 10%）。
為了省時間與記憶體，每類只用 train 前 60 張（再切 80% 記憶／20% 校正）、test 全部，224×224。
特徵存快取（cache/onnx/feat-*.npy），第二次執行很快。延遲因機器而異，講義不貼。

    python V25-onnx-runtime/quantize.py
"""
import logging
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "V24-onnx-export")
from edge_lib import (ONNX_DIR, R18_ONNX, auroc, export_r18, mb, onnx_features, ort_session,  # noqa: E402
                      p50_ms, patch_scores, preprocess, subset)

INT8 = ONNX_DIR / "r18-l23-int8.onnx"
export_r18()
if not INT8.exists():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    logging.getLogger().setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize_dynamic(str(R18_ONNX), str(INT8), weight_type=QuantType.QInt8)

print("== 1. 檔案大小 ==")
print(f"fp32 {mb(R18_ONNX):.1f} MB → int8 {mb(INT8):.1f} MB（{mb(INT8) / mb(R18_ONNX):.2f} 倍）")

rows = []
for cat in ["bottle", "screw"]:
    train, test, y = subset(cat)
    feats = {k: (onnx_features(m, train), onnx_features(m, test)) for k, m in [("fp32", R18_ONNX), ("int8", INT8)]}
    a, b = feats["fp32"][1], feats["int8"][1]
    row = {"類別": cat, "測試張數": len(test), "瑕疵": int(y.sum()),
           "特徵平均相對差": round(float(np.abs(a - b).mean() / np.abs(a).mean()), 3)}
    scores = {}
    for k, (ftr, fte) in feats.items():
        _, scores[k] = patch_scores(ftr, fte)
        row[f"AUROC {k}"] = round(auroc(y, scores[k]), 3)
    row["分數相關"] = round(float(np.corrcoef(scores["fp32"], scores["int8"])[0, 1]), 3)
    rows.append(row)
    del feats, a, b

print("\n== 2. fp32 vs int8：異常偵測 AUROC（train 前 60 張、224×224）==")
t = pd.DataFrame(rows)
print(t.to_string(index=False))
print(f"AUROC 最大變化 {(t['AUROC int8'] - t['AUROC fp32']).abs().max():.3f}")

print("\n== 3. 延遲 p50（毫秒，batch 1、4 執行緒；因機器而異，講義不貼）==")
x1 = preprocess(subset("bottle")[1][:1])
lat = {}
for k, m in [("fp32", R18_ONNX), ("int8", INT8)]:
    s = ort_session(m)
    lat[k] = round(p50_ms(lambda: s.run(None, {"image": x1}), n=15, warmup=3))
print(lat)
print("int8 比 fp32 快：", lat["int8"] < lat["fp32"])
