# V15｜遮罩變成框：準備 YOLO 資料集

MVTec 遮罩 → 連通塊 → 外接矩形 → YOLO 標註格式；讀教師已產生的 `cache/yolo-defects/`（metal_nut＋screw，類別 `defect`，320×320）做統計，不重建。

    python V15-mask-to-yolo/mask_to_box.py
    python V15-mask-to-yolo/dataset_stats.py

資料集由教師用 `PYTHONPATH=. python tools/build_yolo_defects.py` 產生；MVTec AD 為 CC BY-NC-SA 4.0，不隨 repo 散佈。
