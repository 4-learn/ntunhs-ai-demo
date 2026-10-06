# V16｜訓練 YOLO 瑕疵偵測

讀教師訓練好的 `models/defect-yolo11n.pt`：訓練紀錄（`.json`）、每個 epoch 的 loss（`runs/defect/results.csv`）、在 val 54 張上 `model.val()`。

    python V16-yolo-train/read_record.py

訓練由教師先跑（不在課堂上跑）：`PYTHONPATH=. python tools/train_yolo_defects.py`
（yolo11n COCO 預訓練、20 epochs、imgsz 320、batch 8、freeze 10、CPU）。
ultralytics 與 YOLO11 權重為 AGPL-3.0；資料 MVTec AD 為 CC BY-NC-SA 4.0，訓練出的權重僅供課堂使用。
