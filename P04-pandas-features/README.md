# P04｜時間對齊與特徵工程：做出一張事件表

照護：每 10 分鐘檢查點 × 過去心率上升 → 接下來 10 分鐘會不會離床；產線：工件完成前 30 秒振動 → ng。`features.py` 是可重用的步驟。事件表輸出到 /tmp/ntunhs/P04/。

    python P04-pandas-features/events_care.py
    python P04-pandas-features/events_line.py
    python P04-pandas-features/pitfalls.py
