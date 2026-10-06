# V08｜只用良品學正常：像素差 baseline

MVTec 資料夾結構、記憶／校正切割、平均良品像素差、AUROC；`anomaly_lib.py` 是 V08–V13 共用的三種異常分數（A 像素差、B 整張 kNN、C 小格 kNN）與分數快取。平均圖存到 /tmp/ntunhs/V08/。需要 MVTec AD（`MVTEC_ROOT`）。

    python V08-anomaly-intro/pixel_baseline.py
    python V08-anomaly-intro/pitfalls.py
