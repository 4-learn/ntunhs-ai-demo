"""V14｜讀 train.py 存好的權重，在 val 選門檻（20×漏報＋1×誤報），test 只打開一次。

    python V14-defect-finetune/evaluate.py
"""
import json

import numpy as np
import pandas as pd

from defectcls import (CAT, META_FILE, MODEL_FILE, load_images, load_trained, pick_threshold, prob_defect, scores,
                       split_table)

df = split_table()

print(f"== 1. 資料：{CAT} 的瑕疵只在 MVTec test 裡 ==")
t = pd.crosstab([df.source, df.defect], df.split)[["train", "val", "test"]]
print(t.to_string())
for s in ("train", "val", "test"):
    d = df[df.split == s]
    print(f"{s:5s}：{len(d):3d} 張，瑕疵 {d.y.sum():2d}、良品 {(d.y == 0).sum():3d}")

meta = json.loads(META_FILE.read_text(encoding="utf-8"))
print("\n== 2. 訓練紀錄（讀自 models/" + META_FILE.name + "）==")
print(f"起點 {meta['base']}；解凍 features[{meta['unfreeze_from']}:] 與分類頭")
print(f"要訓練的參數 {meta['params_trainable']:,} / {meta['params_total']:,}"
      f"（{meta['params_trainable'] / meta['params_total']:.0%}）")
print(f"{meta['epochs']} epochs、batch {meta['batch']}、lr {meta['lr']}、增強 {meta['augment']}、"
      f"sha256_12 {meta['sha256_12']}")
print("每個 epoch 的 loss：", " ".join(f"{v:.3f}" for v in meta["loss"]))

model = load_trained()
prob = {}
for s in ("val", "test"):
    d = df[df.split == s]
    prob[s] = prob_defect(model, load_images(d.path))

val = df[df.split == "val"]
print("\n== 3. val 掃門檻：成本 = 20×漏報 + 1×誤報 ==")
rows = [{"門檻": th, **scores(val.y, prob["val"], th)} for th in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9)]
print(pd.DataFrame(rows).to_string(index=False))
thr = pick_threshold(val.y, prob["val"])
print(f"val 成本最低的門檻：{thr:.2f}")

test = df[df.split == "test"]
print("\n== 4. test 只打開一次 ==")
rows = [{"model": "全猜良品", **scores(test.y, np.zeros(len(test)), 0.5)},
        {"model": "微調 門檻 0.50", **scores(test.y, prob["test"], 0.5)},
        {"model": f"微調 門檻 {thr:.2f}", **scores(test.y, prob["test"], thr)}]
print(pd.DataFrame(rows).to_string(index=False))

print(f"\n== 5. 錯誤地圖：test 依瑕疵類型（門檻 {thr:.2f}）==")
test = test.assign(prob=prob["test"], pred=(prob["test"] >= thr).astype(int))
g = test.groupby(["source", "defect"]).agg(張數=("y", "size"), 判瑕疵=("pred", "sum"),
                                            prob中位數=("prob", "median"))
print(g.round(3).to_string())
