# P12｜YOLO-pose：一張照片裡的 17 個關鍵點

用 `models/yolo11n-pose.pt` 對 care-room、care-corridor-fall、factory-lifting 推論；拆開 `keypoints.xy`／`.conf`、找低信心點、轉成 poses.csv 的欄位格式；AI 常見錯誤實跑。標註圖與 `photo_keypoints.csv` 存到 `/tmp/ntunhs/P12/`。

    python P12-yolo-pose/pose_infer.py
    python P12-yolo-pose/pitfalls.py

ultralytics 與 YOLO11 權重為 AGPL-3.0，見講義第 2 小節。
