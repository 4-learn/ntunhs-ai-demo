"""V09｜借預訓練模型的眼睛：28×28 小格特徵平均成一個 384 維向量，kNN 距離當異常分數。"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from common import memory_calib_split, mvtec_paths, resnet_patch_features

P = mvtec_paths("bottle")
feats = resnet_patch_features(P["train"])          # 傳完整清單（快取鍵），再用索引切
print("== 1. 預訓練 ResNet18 的特徵 ==")
print(f"bottle train：shape {feats.shape}、dtype {feats.dtype}")
print("意思：每張圖 28×28 個小格，每格 384 個數字（layer2 的 128 ＋ layer3 的 256）")

whole = feats.mean(axis=(1, 2))
print(f"整張平均成一個向量：shape {whole.shape}")
print("前 5 個數字：", [round(float(v), 3) for v in whole[0, :5]])


def run(cat, k=1):
    P = mvtec_paths(cat)
    mem, cal = memory_calib_split(list(range(len(P["train"]))))
    tr = resnet_patch_features(P["train"]).mean(axis=(1, 2))
    te = resnet_patch_features(P["test"]).mean(axis=(1, 2))
    nn = NearestNeighbors(n_neighbors=k).fit(tr[mem])    # 「fit」只是把記憶良品存起來
    s_cal = nn.kneighbors(tr[cal])[0].mean(1)
    s_test = nn.kneighbors(te)[0].mean(1)
    return P["y"], s_cal, s_test, nn


print("\n== 2. bottle：分數＝到最近記憶良品的距離 ==")
y, s_cal, s_test, nn = run("bottle")
tab = pd.DataFrame({"組": ["校正良品", "測試良品", "測試瑕疵"],
                    "平均距離": [s_cal.mean(), s_test[y == 0].mean(), s_test[y == 1].mean()],
                    "最大距離": [s_cal.max(), s_test[y == 0].max(), s_test[y == 1].max()]})
print(tab.to_string(index=False, float_format="%.3f"))
print(f"AUROC {roc_auc_score(y, s_test):.3f}（V08 像素差 0.772）")

print("\n== 3. 沒有 loss、沒有 epoch：「學習」就是存起來 ==")
print(f"記憶裡存了 {nn.n_samples_fit_} 個向量 × {nn.n_features_in_} 維")
print("NearestNeighbors 有可訓練的權重嗎？", hasattr(nn, "coef_"))
print("新增一批良品＝把向量接到記憶後面，不用重新訓練")

print("\n== 4. 門檻＝校正良品最大值：bottle 與 screw ==")
rows = []
for cat in ["bottle", "screw"]:
    y, s_cal, s_test, _ = run(cat)
    pred = s_test > s_cal.max()
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    rows.append({"類別": cat, "AUROC": roc_auc_score(y, s_test), "門檻": s_cal.max(),
                 "recall": tp / y.sum(), "precision": tp / (tp + fp) if tp + fp else 0.0,
                 "漏報": int(y.sum()) - tp, "誤報": fp})
print(pd.DataFrame(rows).set_index("類別").to_string(float_format="%.3f"))

print("\n== 5. k 要取幾個鄰居 ==")
for cat in ["bottle", "screw"]:
    parts = []
    for k in (1, 3, 10):
        y, _, s_test, _ = run(cat, k)
        parts.append(f"k={k} {roc_auc_score(y, s_test):.3f}")
    print(f"{cat}：", "、".join(parts))

print("\n== 6. screw：瑕疵類型的平均名次（1＝最可疑，共 160 張）==")
P = mvtec_paths("screw")
y, _, s_test, _ = run("screw")
rank = pd.Series(s_test).rank(ascending=False)
print(rank.groupby(pd.Series(P["defect"])).mean().sort_values().round(1).to_string())
