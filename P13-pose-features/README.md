# P13｜關鍵點變成表格：姿勢特徵

`common.frame_features` 的 7 個特徵：與畫面大小無關的驗證、三支攝影機比較、NaN 的來源、P12 照片也算成特徵；AI 常寫的像素特徵與 `fillna(0)` 實跑。

    python P13-pose-features/features.py
    python P13-pose-features/pitfalls.py

`photo_keypoints.py` 把 P12 的三張照片推論成 poses.csv 格式（P13 Workshop、P14 會用）。
