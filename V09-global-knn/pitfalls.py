"""V09｜讓 AI 寫，然後挑錯：校正良品自己找到自己、記憶庫混進測試良品、只抽子集的特徵造成重抽。"""
import hashlib

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from common import CACHE, memory_calib_split, mvtec_paths, resnet_patch_features

P = mvtec_paths("screw")
y = P["y"]
tr = resnet_patch_features(P["train"]).mean(axis=(1, 2))
te = resnet_patch_features(P["test"]).mean(axis=(1, 2))
mem, cal = memory_calib_split(list(range(len(P["train"]))))


def evaluate(name, bank, cal_feats, test_feats):
    nn = NearestNeighbors(n_neighbors=1).fit(bank)
    s_cal = nn.kneighbors(cal_feats)[0][:, 0]
    s_test = nn.kneighbors(test_feats)[0][:, 0]
    pred = s_test > s_cal.max()
    good = y == 0
    print(f"{name}：門檻 {s_cal.max():.3f}｜AUROC {roc_auc_score(y, s_test):.3f}｜"
          f"測試良品分數 0 的有 {int((s_test[good] < 1e-6).sum())} 張｜誤報 {int((pred & good).sum())}｜"
          f"recall {(pred & (y == 1)).sum() / y.sum():.3f}")


print("== 錯誤 1：把全部 train 都當記憶，再拿其中一部分當校正 ==")
evaluate("AI 版", tr, tr[cal], te)
evaluate("正確版", tr[mem], tr[cal], te)

print("\n== 錯誤 2：「多放一點良品比較準」——把測試良品也丟進記憶 ==")
bank = np.concatenate([tr[mem], te[y == 0]])
evaluate("AI 版", bank, tr[cal], te)

print("\n== 錯誤 3：只抽記憶子集的特徵 ==")
sub = [P["train"][i] for i in mem]
key = hashlib.sha256(("\n".join(sub) + "|224").encode()).hexdigest()[:16]
print(f"resnet_patch_features(記憶 {len(sub)} 張) 的快取存在嗎？", (CACHE / f"r18-l23-{key}.npy").exists())
print("→ 換一份清單就是另一個快取鍵，會重新跑 ResNet；正確做法是傳完整 P['train'] 再用索引切")
