"""V08｜只用良品學正常：MVTec 資料夾結構、記憶／校正切割、平均良品像素差、AUROC。"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from common import memory_calib_split, mvtec_paths

OUT = Path("/tmp/ntunhs/V08")
OUT.mkdir(parents=True, exist_ok=True)


def gray(path, size=128):
    """灰階、縮成 128×128、0～1。"""
    return np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.float32) / 255


print("== 1. MVTec 資料夾：train 只有良品 ==")
for cat in ["bottle", "screw"]:
    P = mvtec_paths(cat)
    counts = pd.Series(P["defect"]).value_counts().sort_index()
    print(f"{cat}：train/good {len(P['train'])} 張；test {len(P['test'])} 張 →",
          "、".join(f"{k} {v}" for k, v in counts.items()))

results = {}
for cat in ["bottle", "screw"]:
    P = mvtec_paths(cat)
    memory, calib = memory_calib_split(P["train"])
    y = P["y"]
    print(f"\n== 2. {cat}：記憶／校正切割（seed 0）==")
    print(f"記憶 {len(memory)} 張（建立「正常」）、校正 {len(calib)} 張（定門檻）、測試 {len(y)} 張（瑕疵 {y.sum()}）")
    print("記憶與校正重疊：", len(set(memory) & set(calib)), "張")

    mean_img = np.mean([gray(p) for p in memory], axis=0)
    Image.fromarray((mean_img * 255).astype(np.uint8)).save(OUT / f"{cat}-mean.png")

    def score(paths):
        return np.array([np.abs(gray(p) - mean_img).mean() for p in paths])

    s_cal, s_test = score(calib), score(P["test"])
    print(f"\n== 3. {cat}：分數＝與「平均良品」的平均像素差 ==")
    tab = pd.DataFrame({"組": ["校正良品", "測試良品", "測試瑕疵"],
                        "張數": [len(s_cal), int((y == 0).sum()), int(y.sum())],
                        "平均分數": [s_cal.mean(), s_test[y == 0].mean(), s_test[y == 1].mean()],
                        "最大分數": [s_cal.max(), s_test[y == 0].max(), s_test[y == 1].max()]})
    print(tab.round(3).to_string(index=False))

    # AUROC：隨便抽一張瑕疵、一張良品，瑕疵分數比較高的機率
    pos, neg = s_test[y == 1], s_test[y == 0]
    pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    auc = roc_auc_score(y, s_test)
    print(f"\n== 4. {cat}：AUROC ==")
    print(f"瑕疵 {len(pos)} × 良品 {len(neg)} = {len(pos) * len(neg)} 對，瑕疵分數較高的比例 {pairs:.3f}")
    print(f"roc_auc_score(y, 分數) = {auc:.3f}")

    thr = s_cal.max()
    pred = s_test > thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn = int(y.sum()) - tp
    results[cat] = dict(AUROC=auc, 門檻=thr, recall=tp / (tp + fn),
                        precision=tp / (tp + fp) if tp + fp else 0.0, 漏報=fn, 誤報=fp)

print("\n== 5. 門檻＝校正良品分數的最大值 ==")
print(pd.DataFrame(results).T.astype({"漏報": int, "誤報": int}).to_string(float_format="%.3f"))

P = mvtec_paths("screw")
memory, _ = memory_calib_split(P["train"])
mean_img = np.mean([gray(p) for p in memory], axis=0)
one = gray(memory[0])
print("\n== 6. 為什麼 screw 的平均良品沒用 ==")
print(f"bottle 平均良品的像素標準差 {np.mean([gray(p) for p in memory_calib_split(mvtec_paths('bottle')['train'])[0]], axis=0).std():.3f}")
print(f"screw 平均良品的像素標準差 {mean_img.std():.3f}；單張 screw 良品 {one.std():.3f}")
print("（平均圖存在 /tmp/ntunhs/V08/，螺絲方向每張不同，平均起來糊成一片）")
