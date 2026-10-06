# V14｜有標註才能做的事：瑕疵分類微調

MVTec metal_nut 良品／瑕疵二分類：MobileNetV3-Small 解凍最後幾個區塊微調（CPU、8 epochs）。`train.py` 由教師先跑一次，權重存到 `models/defect-cls-mnv3s.pt`（＋ `.json` 紀錄）；`evaluate.py` 讀權重、在 val 用 20×漏報＋1×誤報選門檻、test 只開一次；`pitfalls.py` 是挑錯題實測。

    python V14-defect-finetune/train.py      # 教師先跑（多人共用 CPU 約 4–5 分鐘）
    python V14-defect-finetune/evaluate.py
    python V14-defect-finetune/pitfalls.py

MVTec AD 為 CC BY-NC-SA 4.0，不隨 repo 散佈；權重檔以它訓練，同樣僅供課堂使用。
