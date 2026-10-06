# P15｜時間窗口：跌倒還是坐下

單格分類分不出 fall／lie_bed／sit_down；`common.window_features`（過去 1 秒的變化量）；序列層級 `is_fall_seq`（序列分數＝每格機率最大值）；困難負例。AI 常見錯誤（窗口跨序列、偷看未來、以格計分）實跑。

    python P15-pose-window/window.py
    python P15-pose-window/pitfalls.py
