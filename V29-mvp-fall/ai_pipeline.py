"""V29「讓 AI 寫，然後挑錯」：AI 寫的跌倒管線常見的三個錯，各跑一次給你看。

    python V29-mvp-fall/ai_pipeline.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import shutil  # noqa: E402
import sys  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from common import ROOT, WINDOW_FEATURES, load_poses, subject_split, window_features  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fall_pipeline import load_models, run_night, schedule, score_batch  # noqa: E402
from deploy import load_config  # noqa: E402
from event_api import create_app, make_client  # noqa: E402

WORK = Path("/tmp/ntunhs/V29/ai")
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
raw = yaml.safe_load((ROOT / "V32-deploy-checklist/config/fall.yaml").read_text(encoding="utf-8"))
raw["log_path"] = str(WORK / "log.jsonl")
(WORK / "fall.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
cfg = load_config(WORK / "fall.yaml")
fall_model, pose_model = load_models(cfg)
_, test = subject_split(load_poses())
plan = score_batch(schedule(test), fall_model, pose_model)

print("== AI 錯誤 1：evidence 寫 {\"frame\": 40} ==")
client = make_client(create_app())
ev = {"event_id": "s1cam1-20260112-220004-0", "source": "pose", "site": "松年本館", "camera_id": "cam1",
      "event_time": "2026-01-12T22:00:04+08:00", "kind": "fall", "score": 0.993, "threshold": 0.95,
      "evidence": {"seq_id": "SQ0291", "frame": 40},
      "model": {"name": "fall-hgb", "version": "2026.01", "sha256_12": "05ba1c2d31cf"}}
r = client.post("/events", json=ev)
print(f"POST /events → {r.status_code}：{r.json()['detail'][0]['msg']}")
ev["event_id"] = "s1cam1-20260112-220004-1"
ev["evidence"] = {"seq_id": "SQ0291", "frame_no": 40}
print(f"改成 frame_no → {client.post('/events', json=ev).status_code}")

print("\n== AI 錯誤 2：串流緩衝只留 10 格（WINDOW = 10，所以 deque(maxlen=10)）==")
one = plan[plan["seq_id"] == "SQ0263"]
batch = one["p_fall"].to_numpy()
for n in (10, 11):
    buf, ps = deque(maxlen=n), []
    for i in range(len(one)):
        buf.append(one.iloc[[i]])
        w = pd.concat(list(buf), ignore_index=True)
        ps.append(fall_model.predict_proba(window_features(w)[WINDOW_FEATURES].tail(1))[0, 1])
    ps = np.array(ps)
    cross = int(((ps >= cfg.threshold.value) != (batch >= cfg.threshold.value)).sum())
    print(f"緩衝 {n} 格：與整批版最大差異 {np.abs(ps - batch).max():.3f}，過門檻與否不同的格 {cross}")

print("\n== AI 錯誤 3：「門檻 0.95 已經很高，狀態機可以拿掉」==")
sys.path.insert(0, str(ROOT / "V21-fall-state-machine"))
from fsm import FallFSM  # noqa: E402

y = plan.groupby("seq_id")["is_fall_seq"].first()
rows = []
for t in (0.95, 0.5):
    for name in ("V21 狀態機", "直接送"):
        hit = {}
        for sid, d in plan.groupby("seq_id"):
            if name == "直接送":
                hit[sid] = bool((d["p_fall"] >= t).any())
            else:
                f = FallFSM(threshold=t)
                hit[sid] = "alert" in [f.step(p, g) for p, g in zip(d["p_fall"], d["pose_pred"])]
        h = pd.Series(hit)
        rows.append({"門檻": t, "做法": name, "抓到跌倒": f"{int(h[y == 1].sum())}/11", "誤報": int(h[y == 0].sum())})
print(pd.DataFrame(rows).set_index(["門檻", "做法"]).to_string())
print("（以序列為單位；同一批測試序列）")
