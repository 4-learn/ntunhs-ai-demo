"""V30｜模擬「上線一週」：兩組的事件與值班人員的確認（verdict）。

全部是程式產生（固定 seed），不是真實機構或產線的紀錄。V30 的講義與 workshop 共用。

    from sim_week import simulate_week
    events, acks = simulate_week(seed=0)     # events：Event JSON 清單；acks：{event_id: (by, verdict, ack_time)}

設計（讓學員在統計裡看得到的現象）：
- 健康組：本館 3 支攝影機。cam1（寢室）常把「本來就躺著、翻身」誤判成跌倒（V19 的 lie_bed 困難負例），
  cam2（走廊）、cam3（交誼廳）誤報少。真跌倒一週約 5 次。
- 製造組：line1cam1，日班 08–17 點，螺絲。真瑕疵約千分之一的量級，誤報大多貼著門檻。
- 約 1 成事件沒有人按確認（夜班忙）。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

TW = timezone(timedelta(hours=8))
START = datetime(2026, 1, 12, 0, 0, tzinfo=TW)          # 週一 00:00
DAYS = 7

FALL_MODEL = {"name": "fall-hgb", "version": "2026.01", "sha256_12": "05ba1c2d31cf"}
NG_MODEL = {"name": "patch-knn-screw", "version": "2026.01", "sha256_12": "a41d6e90c3b8"}
FALL_THRESHOLD = 0.95  # P16 定版（demo/config/fall.yaml）
NG_THRESHOLD = 0.62

# 每支攝影機一週的（真事件數、誤報數）期望值
FALL_CAMS = {"cam1": (2, 34), "cam2": (2, 9), "cam3": (1, 6)}
NG_CAM = ("line1cam1", 9, 30)
NIGHT_NURSES = ["N02", "N05"]
DAY_NURSES = ["N01", "N03", "N04", "N06"]
QC = ["Q01", "Q02"]


def _score(rng, true, thr):
    """真事件分數偏高、誤報分數貼著門檻（都 ≥ 門檻，因為只有過門檻的才會變成事件）。"""
    b = rng.beta(4.0, 1.5) if true else rng.beta(1.2, 4.0)
    return round(thr + (1 - thr) * b, 3)


def _fall_time(rng, true):
    day = int(rng.integers(DAYS))
    # 跌倒與誤報都偏夜間（起身如廁、翻身）；分鐘、秒隨機
    hour = int(rng.choice(24, p=_hour_p()))
    return START + timedelta(days=day, hours=hour, minutes=int(rng.integers(60)), seconds=int(rng.integers(60)))


def _hour_p():
    w = np.ones(24)
    w[[0, 1, 2, 3, 4, 5, 22, 23]] = 3.0
    return w / w.sum()


def simulate_week(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []                                   # (event, truth)
    for cam, (n_true, n_false) in FALL_CAMS.items():
        for truth, n in ((True, rng.poisson(n_true)), (False, rng.poisson(n_false))):
            for _ in range(n):
                t = _fall_time(rng, truth)
                rows.append(({"source": "pose", "site": "松年本館", "camera_id": cam, "event_time": t,
                              "kind": "fall", "score": _score(rng, truth, FALL_THRESHOLD),
                              "threshold": FALL_THRESHOLD,
                              "evidence": {"seq_id": f"LIVE{int(rng.integers(10000)):04d}", "keypoints_only": True},
                              "model": FALL_MODEL}, truth))
    cam, n_true, n_false = NG_CAM
    for truth, n in ((True, rng.poisson(n_true)), (False, rng.poisson(n_false))):
        for _ in range(n):
            day = int(rng.integers(5))          # 週一到週五
            t = START + timedelta(days=day, hours=int(rng.integers(8, 17)), minutes=int(rng.integers(60)),
                                  seconds=int(rng.integers(60)))
            rows.append(({"source": "defect", "site": "精工五金", "camera_id": cam, "event_time": t,
                          "kind": "ng", "score": _score(rng, truth, NG_THRESHOLD), "threshold": NG_THRESHOLD,
                          "evidence": {"part_id": f"P{int(rng.integers(100000)):05d}", "category": "screw",
                                       "heatmap_max_xy": [int(rng.integers(224)), int(rng.integers(224))]},
                          "model": NG_MODEL}, truth))
    rows.sort(key=lambda r: (r[0]["event_time"], r[0]["camera_id"]))

    events, acks, seen = [], {}, {}
    for ev, truth in rows:
        t = ev["event_time"]
        prefix = ("s1" + ev["camera_id"]) if ev["source"] == "pose" else ev["camera_id"]
        key = (prefix, t.replace(microsecond=0))
        n = seen.get(key, 0)
        seen[key] = n + 1
        ev = {**ev, "event_id": f"{prefix}-{t:%Y%m%d-%H%M%S}-{n}", "event_time": t.isoformat()}
        events.append(ev)
        night = t.hour < 7 or t.hour >= 22
        if rng.random() < (0.15 if night else 0.05):   # 沒人按確認
            continue
        if ev["source"] == "pose":
            by = str(rng.choice(NIGHT_NURSES if night else DAY_NURSES))
        else:
            by = str(rng.choice(QC))
        # 人也會判錯：真的說成誤報 2%、誤報說成真的 3%
        flip = rng.random() < (0.02 if truth else 0.03)
        verdict = "true_alarm" if (truth != flip) else "false_alarm"
        # 到場確認要幾分鐘：白天人多 1–4 分；夜班 2–11 分（V30 第 4 小節的升級推播用）
        lo, hi = (2, 12) if night else (1, 5)
        ack_time = t + timedelta(minutes=int(rng.integers(lo, hi)), seconds=int(rng.integers(60)))
        acks[ev["event_id"]] = (by, verdict, ack_time)
    return events, acks
