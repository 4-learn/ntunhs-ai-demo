"""V29 示範：健康組 MVP 端到端——骨架串流 → 窗口特徵 → P16 跌倒模型 → V21 狀態機＋V22 冷卻 → Event JSON → 事件 API。

    python V29-mvp-fall/mvp_fall.py

測試受試者 S19–S24 的 84 段序列，依攝影機排成「一晚」：每支攝影機每 5 分鐘播一段（再停留 3 秒）。
測試集在 P16 已經打開過一次並記錄；本節不再用它選任何東西，只驗證「接成管線後結果和 P16 相同」。
NTUNHS_MOCK=1：不連 EZCam、不連網路；事件 API 用 TestClient＋SQLite :memory:。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from common import FPS, ROOT, WINDOW_FEATURES, load_poses, subject_split, window_features  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fall_pipeline import POSTURE_PATH, load_models, run_night, schedule  # noqa: E402
from deploy import JsonLog, load_config, sha256_12, start  # noqa: E402
from event_api import Event, create_app, make_client  # noqa: E402
from pipeline import TW  # noqa: E402

pd.set_option("display.unicode.east_asian_width", True)
WORK = Path("/tmp/ntunhs/V29")
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)

print("== 1. 設定與定版檔（V32 設定檔 → P16 定版卡）==")
raw = yaml.safe_load((ROOT / "V32-deploy-checklist/config/fall.yaml").read_text(encoding="utf-8"))
raw["log_path"] = str(WORK / "fall-monitor.jsonl")
(WORK / "fall.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
cfg = load_config(WORK / "fall.yaml")
card = yaml.safe_load((ROOT / cfg.model.card).read_text(encoding="utf-8"))
print(f"跌倒模型 {cfg.model.path}  sha256_12 {cfg.model.sha256_12}  門檻 {cfg.threshold.value}  冷卻 {cfg.cooldown_s:.0f}s")
print(f"P16 定版卡上的核准狀態：{card['threshold']['approved_by']}；V32 設定檔：{cfg.threshold.approved_by} {cfg.threshold.approved_on}")
fall_model, pose_model = load_models(cfg)
print(f"姿勢模型 models/{POSTURE_PATH.name}  sha256_12 {sha256_12(POSTURE_PATH)}（P14 做法，S01–S18 訓練）")
log = JsonLog(cfg.log_path, clock=lambda: datetime(2026, 1, 12, 21, 59, tzinfo=TW))
start(cfg, ROOT, log)
print("啟動自檢全部通過 → 允許啟動")

print("\n== 2. 串流版窗口特徵 vs P16 整批版（差異要是 0）==")
df = load_poses()
_, test = subject_split(df)
plan = schedule(test)
one = plan[plan["seq_id"] == plan[plan["is_fall_seq"] == 1]["seq_id"].iloc[0]]
batch = fall_model.predict_proba(window_features(one)[WINDOW_FEATURES])[:, 1]
from fall_pipeline import FallDetector  # noqa: E402
det = FallDetector(fall_model, pose_model, cfg.threshold.value)
for i in range(len(one)):
    det.predict(one.iloc[[i]])
stream = np.array([r["p"] for r in det.trace])
print(f"{one['seq_id'].iloc[0]}（{len(one)} 格）：緩衝 11 格，逐格機率最大差異 {np.abs(stream - batch).max():.6f}")
print("→ 一致，所以下面整晚用整批算好的機率餵管線（每格仍逐一經過狀態機、冷卻、送出）")

print("\n== 3. 跑一晚：3 支攝影機、84 段序列 ==")
client = make_client(create_app())
seq, pipes, events = run_night(test, cfg, client)
for cam, p in pipes.items():
    st = pd.Series([r["status"] for r in p.log]).value_counts()
    print(f"{cam}：{len(p.log):5d} 格  sent {st.get('sent', 0)}  cooldown {st.get('cooldown', 0)}  below {st.get('below', 0)}")
print(f"事件 API 收到 {len(events)} 筆；每筆都通過 Event 驗證："
      f"{all(Event.model_validate({k: v for k, v in e.items() if k != 'ack'}) for e in events)}")
print(f"evidence 只有骨架序列編號：{sorted({k for e in events for k in e['evidence']})}")

print("\n== 4. 對答案（以序列為單位，和 P16 同一把尺）==")
tp = int(((seq.y == 1) & (seq.alarm == 1)).sum())
fn = int(((seq.y == 1) & (seq.alarm == 0)).sum())
fp = int(((seq.y == 0) & (seq.alarm == 1)).sum())
print(f"跌倒 {int(seq.y.sum())} 段：抓到 {tp}、漏報 {fn}；非跌倒 {int((seq.y == 0).sum())} 段：誤報 {fp}")
print(f"precision {tp / max(tp + fp, 1):.3f}  recall {tp / max(tp + fn, 1):.3f}  成本 10×{fn}＋{fp} = {10 * fn + fp}")
print(f"P16 測試集（config/fall.yaml）：tp {card['test']['tp']}、fn {card['test']['fn']}、fp {card['test']['fp']}"
      f"、precision {card['test']['precision']}、recall {card['test']['recall']}")
print("誤報：")
print(seq[(seq.y == 0) & (seq.alarm == 1)][["camera_id", "activity", "score"]].to_string())

print("\n== 5. 從跌倒開始到通知：要等多久 ==")
f = seq[(seq.y == 1) & (seq.alarm == 1)]
delay = (f["alert_frame"] - f["fall_frame"]) / FPS
print(f"跌倒開始 → 事件送出：中位數 {delay.median():.1f} 秒（最短 {delay.min():.1f}、最長 {delay.max():.1f}）")
print("組成：跌倒過程約 1 秒＋狀態機要求躺／坐地連續 2 秒")

print("\n== 6. 這一晚，每支攝影機叫了幾次 ==")
ev = pd.DataFrame([{"cam": e["camera_id"], "seq_id": e["evidence"]["seq_id"]} for e in events])
ev["真跌倒"] = ev["seq_id"].map(seq["y"])
tab = ev.groupby("cam").agg(通知=("seq_id", "size"), 真跌倒=("真跌倒", "sum"))
tab["誤報"] = tab["通知"] - tab["真跌倒"]
print(tab.to_string())
hours = (plan["t_wall"].max() - plan["t_wall"].min()).total_seconds() / 3600
print(f"模擬時段 {hours:.1f} 小時；84 段裡有 11 段跌倒（比例遠高於真實夜晚，見 V22）")

print("\n== 7. 一筆事件 ==")
e = events[0]
print(json.dumps({k: e[k] for k in ("event_id", "camera_id", "event_time", "score", "threshold", "evidence", "model")},
                 ensure_ascii=False))
