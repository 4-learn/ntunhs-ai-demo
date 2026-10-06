# V17｜偵測評估：IoU、漏報與誤報

自己寫 IoU 與逐框配對（`detlib.py`），在 V15 的 test 56 張上算框層級 TP/FP/FN；
再改成影像層級（任何框信心 ≥ 門檻即 ng），在 val 上以 20×漏報＋1×誤報選門檻，test 只開一次。

    python V17-detection-eval/evaluate.py

偵測結果快取在 `cache/det-<權重sha>-<split>-0.01.json`。可用環境變數 `DEFECT_WEIGHTS` 換權重比較。
