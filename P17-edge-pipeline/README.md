# P17｜EZCam → Pi 5 → 骨架 → 事件

邊緣端管線：EZCam 擷取（mock）→ YOLO-pose → 51 欄骨架 → 1 秒窗口緩衝 → P16 定版跌倒模型 → 簡化版 Event JSON。影像用完即丟，只往下傳骨架。

    NTUNHS_MOCK=1 python P17-edge-pipeline/pipeline.py
    python P17-edge-pipeline/bench.py      # 本機各階段耗時（因機器而異；Pi 5 未實測）

`edge_pipeline.py` 是零件（`FallDetector`、`yolo_rows`、`load_fall_model`），需要先執行 `P16-pose-eval/build_fall_model.py` 產生 `models/fall-hgb.joblib` 與 `config/fall.yaml`。真機模式需要環境變數 `EZCAM_API_KEY`（不可寫進檔案），見 `P09-ezcam/ezcam_client.py`。
