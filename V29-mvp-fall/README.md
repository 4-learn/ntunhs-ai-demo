# V29｜健康組 MVP：跌倒監測管線

骨架串流（S19–S24，mock）→ 窗口特徵 → P16 定版跌倒模型（`models/fall-hgb.joblib`）→ V21 狀態機＋V22 冷卻 → Event JSON → V27 事件 API。

    python V29-mvp-fall/mvp_fall.py      # 端到端一晚，對答案（約 20 秒）
    python V29-mvp-fall/ai_pipeline.py   # AI 常見錯

`fall_pipeline.py` 是元件（V33–V35 也用）。第一次執行會建 `models/posture-hgb.joblib`（P14 做法的姿勢模型，S01–S18 訓練，單執行緒可重現）。
設定檔讀 `V32-deploy-checklist/config/fall.yaml`（其 model.card 指向 P16 的 `config/fall.yaml`）。
