# P09｜EZCam 實拍：從攝影機到陣列

EZCam client（要求拍照 → 輪詢 /api/latest → 下載 JPEG → numpy）；320×240 的像素預算；3 秒輪詢不是串流，對跌倒偵測的影響；AI 常見錯誤（離線模擬）。

`NTUNHS_MOCK=1` 時讀 `data/photos/ezcam-sample.jpg`、不連網路。真實模式需要環境變數 `EZCAM_API_KEY`（**不可寫進任何檔案**），可選 `EZCAM_BASE_URL`。

    NTUNHS_MOCK=1 python P09-ezcam/ezcam_client.py
    NTUNHS_MOCK=1 python P09-ezcam/ezcam_frames.py
    NTUNHS_MOCK=1 python P09-ezcam/pitfalls.py
