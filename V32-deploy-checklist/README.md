# V32｜上線清單：設定、版本、紀錄

`deploy.py`：YAML 設定檔（pydantic 驗證）、模型檔 sha256_12、JSON Lines 紀錄、啟動自檢（hash 不符拒絕啟動）、版本切換與回滾。`config/` 是兩組的設定檔範例。示範只寫入 `/tmp/ntunhs/V32/`。

    python V32-deploy-checklist/checklist.py
