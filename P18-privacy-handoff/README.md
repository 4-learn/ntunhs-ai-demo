# P18｜只存骨架不存畫面：隱私設計與交接

資料最小化（影像 vs 骨架的資料量、事件內容）、骨架能不能認出是誰（實測），以及給影像 AI 課的交接包。

    python P18-privacy-handoff/privacy.py
    python P18-privacy-handoff/handoff.py     # 寫出 handoff/fall-handoff.json

`handoff.py` 需要先執行 `P16-pose-eval/build_fall_model.py`。交接包內容：資料與評估集清單的 SHA-256、標註規則、已知標註錯誤、模型 hash、門檻與依據、驗證與測試分數、每段測試序列的分數、待辦事項；不含任何影像。
