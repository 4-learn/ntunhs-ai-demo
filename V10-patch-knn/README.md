# V10｜切成小格比：PatchCore 簡化版

小格記憶庫（記憶良品的 28×28 小格抽 10%，rng 0）、最不像的小格＝整張分數、記憶庫 1%／10%／100% 的取捨。共用實作在 `V08-anomaly-intro/anomaly_lib.py`。第一次執行要算 100% 記憶庫（screw 約 10 分鐘以上），之後讀 demo/cache/scores-*.npz。

    python V10-patch-knn/patch_knn.py
    python V10-patch-knn/pitfalls.py
