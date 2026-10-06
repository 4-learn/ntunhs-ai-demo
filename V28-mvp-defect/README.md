# V28｜製造組 MVP：瑕疵檢測管線

輸送帶取像（MVTec 測試影像依序）→ 小格 kNN（`common.resnet_patch_features` 快取＋V12 成本門檻）→ Event JSON → V27 事件 API，算漏報與誤報。`defect_model.py` 供 V33–V35 import；記憶庫寫在 `cache/models/`（MVTec 衍生物，不進 repo）。

    python V28-mvp-defect/mvp_defect.py            # metal_nut；加參數 screw 跑螺絲
    python V28-mvp-defect/ai_pipeline.py
