# V12｜門檻：校正良品、少量瑕疵與代價

定版驗證切割（每類 test 瑕疵固定抽 10 張，seed 0）、在驗證集上用 20×漏報＋1×誤報選門檻、與校正最大值比較；挑錯題示範用測試集選門檻。讀 V10 的分數快取。

    python V12-anomaly-threshold/threshold.py
    python V12-anomaly-threshold/pitfalls.py
