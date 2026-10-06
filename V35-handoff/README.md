# V35｜交接包

    python V35-handoff/handoff.py    # 寫出 handoff/{defect,fall}/（handoff.json、eval-list.csv、SHA256SUMS）並驗證

格式沿用 P18（`P18-privacy-handoff/handoff/fall-handoff.json`）的 11 個區塊，另加 models、known_model_errors、acceptance（健康組再加 pipeline）。
`handoff_lib.verify(dir)`：接手方的驗證（檔案、資料、評估集、模型 hash、測試分數、門檻與部署設定）。
製造組不含 MVTec 影像或記憶庫，只有相對路徑與 hash。
