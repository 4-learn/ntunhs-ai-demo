# V18｜有標註 vs 只有良品：怎麼選

同一批影像（V15 的 test 56 張：metal_nut＋screw）上，比較 YOLO（V16，有標註）與 V10 小格 kNN（只有良品）：
影像層級 recall／precision／成本、依瑕疵類型、定位是否落在標註框內。

    python V18-supervised-vs-anomaly/compare.py

小格 kNN 用 `V08-anomaly-intro/anomaly_lib.py` 的 `scores_patch`（分數有快取）與 `cost_threshold`。
