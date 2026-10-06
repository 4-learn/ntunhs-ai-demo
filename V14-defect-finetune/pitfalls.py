"""V14｜讓 AI 寫，然後挑錯：只用 MVTec test 隨機切、加水平翻轉增強、只報 accuracy。

    python V14-defect-finetune/pitfalls.py
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from common import mvtec_paths
from defectcls import CAT, load_images, load_trained, prob_defect, split_table

print("== 錯誤 1：只用 MVTec test 切 80/20，測試集裡有幾張良品？ ==")
P = mvtec_paths(CAT)
_, te_idx = train_test_split(np.arange(len(P["test"])), test_size=0.2, random_state=0)
n_good = int((P["y"][te_idx] == 0).sum())
print(f"MVTec test 共 {len(P['test'])} 張（良品 {(P['y'] == 0).sum()}）；AI 的測試集 {len(te_idx)} 張，良品只有 {n_good} 張")
print(f"→ 「誤報 0 次」最多只代表 0/{n_good}，量不出誤報率；本節另從 train/good 補良品")

print("\n== 錯誤 2：水平翻轉增強——把良品翻面，正好就是 flip 瑕疵 ==")
df = split_table()
model = load_trained()
good = df[(df.split == "test") & (df.y == 0)]
X = load_images(good.path)
p0 = prob_defect(model, X)
p1 = prob_defect(model, torch.flip(X, dims=(3,)))
print(f"test 良品 {len(good)} 張：原圖平均瑕疵機率 {p0.mean():.3f}；水平翻轉後 {p1.mean():.3f}，"
      f"判成瑕疵（≥0.5）{int((p1 >= 0.5).sum())} 張")
print("→ 這個模型主要靠「背面的表面」認 flip，鏡像良品多半仍判良品；但鏡像良品在產線上不存在，標成良品就是錯標")

print("\n== 錯誤 3：只報 accuracy ==")
test = df[df.split == "test"]
acc_base = (test.y == 0).mean()
print(f"test {len(test)} 張、瑕疵 {test.y.sum()} 張：全猜良品的 accuracy 就有 {acc_base:.3f}，漏報 {test.y.sum()}")
