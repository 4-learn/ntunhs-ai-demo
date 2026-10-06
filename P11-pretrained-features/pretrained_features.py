"""P11｜預訓練 CNN 在看什麼：ResNet18 整張圖 512 維特徵、良品與瑕疵的距離、最近鄰。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import mvtec_paths                       # noqa: E402
from features import WEIGHTS, load_model, preprocess, resnet_vectors   # noqa: E402

B, S = mvtec_paths("bottle"), mvtec_paths("screw")
labels = WEIGHTS.meta["categories"]

print("== 1. 原本的 ResNet18：認 1000 類東西 ==")
model = load_model()
with torch.no_grad():
    for name, p in [("bottle 良品", B["train"][0]), ("screw 良品", S["train"][0])]:
        prob = model(preprocess(Image.open(p).convert("RGB"))[None]).softmax(1)[0]
        top = prob.topk(3)
        print(f"{name}：", "、".join(f"{labels[i]} {v:.2f}" for v, i in zip(top.values, top.indices)))
print("ImageNet 沒有「良品／瑕疵」這種類別——我們不要它最後的答案，要它中間的「看法」")

print("\n== 2. 拿掉最後一層：每張圖 → 512 個數字 ==")
btr, bte = resnet_vectors(B["train"]), resnet_vectors(B["test"])
str_, ste = resnet_vectors(S["train"]), resnet_vectors(S["test"])
print(f"bottle 良品 {btr.shape}；screw 良品 {str_.shape}")
print("第 1 張 bottle 的前 6 個數：", [round(float(x), 3) for x in btr[0, :6]])
print(f"最小值 {btr.min():.3f}（ReLU 之後的平均，不會是負的）；向量長度平均 {np.linalg.norm(btr, axis=1).mean():.1f}")


def dist(a, b):
    """a 的每一列到 b 的每一列的歐氏距離，shape (len(a), len(b))。"""
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


print("\n== 3. 距離：同類近、不同類遠 ==")
bb = dist(btr, btr)[np.triu_indices(len(btr), 1)]
ss = dist(str_, str_)[np.triu_indices(len(str_), 1)]
print(f"bottle 良品 ↔ bottle 良品  平均距離 {bb.mean():5.2f}")
print(f"screw  良品 ↔ screw  良品  平均距離 {ss.mean():5.2f}")
print(f"bottle 良品 ↔ screw  良品  平均距離 {dist(btr, str_).mean():5.2f}")
print("512 個數字裡，「這是什麼東西」占了很大的分量")

print("\n== 4. 每張測試圖到「最近的良品」多遠 ==")


def nn_table(tr, te, P):
    d = dist(te, tr)
    df = pd.DataFrame({"類型": P["defect"], "最近距離": d.min(1), "最近的良品": d.argmin(1)})
    return df, df.groupby("類型")["最近距離"].agg(["count", "mean", "min", "max"])


bdf, btab = nn_table(btr, bte, B)
print("bottle：")
print(btab.round(2).to_string())
sdf, stab = nn_table(str_, ste, S)
print("screw：")
print(stab.round(2).to_string())

print("\n== 5. 用一個門檻分得開嗎？ ==")
for name, df in [("bottle", bdf), ("screw", sdf)]:
    g = df.loc[df["類型"] == "good", "最近距離"]
    bad = df.loc[df["類型"] != "good", "最近距離"]
    print(f"{name}：測試良品最遠 {g.max():.2f}；瑕疵中比它還近的 {(bad <= g.max()).sum()}/{len(bad)} 張"
          f"；瑕疵比良品平均遠 {bad.mean() / g.mean():.2f} 倍")

print("\n== 6. 最近鄰：瑕疵圖最像哪一張良品 ==")
for name, df, P in [("bottle", bdf, B), ("screw", sdf, S)]:
    for t in ["good", df.loc[df["類型"] != "good"].sort_values("最近距離").iloc[0]["類型"]]:
        r = df[df["類型"] == t].sort_values("最近距離").iloc[0]
        print(f"{name} {t:<15} {Path(P['test'][r.name]).name} → 良品 #{r['最近的良品']:<3} 距離 {r['最近距離']:.2f}")

print("\n== 7. screw 為什麼難：同一顆螺絲轉 90 度 ==")
g0 = Image.open(S["test"][S["defect"].index("good")]).convert("RGB")
with torch.no_grad():
    fc, model.fc = model.fc, torch.nn.Identity()
    v = model(torch.stack([preprocess(g0), preprocess(g0.rotate(90))])).numpy()
    model.fc = fc
print(f"原圖 ↔ 轉 90 度        距離 {np.linalg.norm(v[0] - v[1]):.2f}")
print(f"screw 瑕疵到最近良品    平均 {sdf.loc[sdf['類型'] != 'good', '最近距離'].mean():.2f}")
print("同一顆螺絲只是轉個角度，距離就比「有瑕疵」大兩倍多：整張圖的向量對擺放角度很敏感")
print(f"screw 良品彼此平均距離 {ss.mean():.2f} 也是這個原因——{len(str_)} 張良品的角度各不相同")
