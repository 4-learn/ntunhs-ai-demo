# V21｜跌倒狀態機：規則與模型合作

站立 →（窗口模型機率 ≥ 門檻）→ 疑似跌倒 →（躺／坐地持續 ≥ 2 秒）→ 告警。`fsm.py` 是狀態機本體（V22、V23 也用），`state_machine.py` 在驗證集 S15–S18 上和「單格模型直接告警」比較。

```bash
python V21-fall-state-machine/state_machine.py
```
