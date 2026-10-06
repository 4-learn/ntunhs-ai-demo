# V19｜從單張到串流：骨架序列

把 `poses.csv` 的序列逐格播放成串流（generator），並用降採樣實驗量化「每 3 秒拍一張」會漏掉多少跌倒。`fallmodel.py` 是 V19–V23 共用的小模型（只用 S01–S18 訓練）。

```bash
python V19-pose-stream/stream.py
python V19-pose-stream/sample_rate.py
```
