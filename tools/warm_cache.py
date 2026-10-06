"""預先算好 6 類 MVTec 的 ResNet18 小格特徵快取（教師開課前跑一次）。"""
import common as C

for c in C.MVTEC_CATEGORIES:
    P = C.mvtec_paths(c)
    for split in ("train", "test"):
        a = C.resnet_patch_features(P[split])
        print(c, split, a.shape)
