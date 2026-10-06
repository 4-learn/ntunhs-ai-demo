# V09｜借預訓練模型的眼睛：整張特徵 kNN

ResNet18 小格特徵（`common.resnet_patch_features`，讀 demo/cache/）平均成每張圖一個 384 維向量，到最近記憶良品的距離＝異常分數；挑錯題示範校正／測試良品混進記憶。

    python V09-global-knn/global_knn.py
    python V09-global-knn/pitfalls.py
