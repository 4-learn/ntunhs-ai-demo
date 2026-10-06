"""V22：把驗證集序列串成「一整晚」的模擬時間軸，算林護理長最在意的數字：每晚會被叫幾次。

去抖動（連續告警合併成一件）＋同一人冷卻 60 秒。訓練 S01–S14、驗證 S15–S18；測試集不打開。

    python V22-debounce/night.py
"""
import sys

import numpy as np
import pandas as pd

from common import FPS, ROOT, load_poses, subject_split

sys.path.insert(0, str(ROOT / "V19-pose-stream"))
sys.path.insert(0, str(ROOT / "V21-fall-state-machine"))
from fallmodel import FIT, VALID, prepare, score_frames, train_models  # noqa: E402
from fsm import FallFSM  # noqa: E402

THRESHOLD = 0.5
NIGHT_H = 8          # 22:00–06:00
SESSIONS = 10        # 每位住民每晚在鏡頭前出現幾次（假設）
PER_SESSION = 6      # 每次連續做幾件事（每件 4 秒，前後相接）
ON_FLOOR_S = 60      # 跌倒後到護理師趕到前，人停在地上的秒數（假設）
MERGE_S = 1.0        # 旗標間隔小於 1 秒就算同一件
COOLDOWN_S = 60.0    # 同一人通知後 60 秒內不再通知


def extend_on_floor(d: pd.DataFrame, seconds: float) -> pd.DataFrame:
    """跌倒序列後面接「躺在地上」：重複播放最後 1 秒（保留真實的抖動與漏點）。"""
    tail = d.tail(FPS)
    n = int(seconds * FPS)
    extra = pd.concat([tail] * (n // FPS + 1)).head(n).copy()
    extra["frame"] = d["frame"].max() + 1 + np.arange(n)
    return pd.concat([d, extra])


def scored_episodes(val, win, pose):
    """每段序列先算好逐格的窗口機率與姿勢（各晚重複使用）。跌倒序列後面接 60 秒倒地。"""
    parts = [extend_on_floor(d, ON_FLOOR_S) if d["is_fall_seq"].iloc[0] == 1 else d
             for _, d in val.groupby("seq_id")]
    v = score_frames(prepare(pd.concat(parts)), win, pose)
    return {sid: dict(subject=d["subject_id"].iloc[0], activity=d["activity"].iloc[0],
                      fall=int(d["is_fall_seq"].iloc[0]), p=d["p_fall"].to_numpy(),
                      pose=d["pose_pred"].to_numpy())
            for sid, d in v.groupby("seq_id")}


def night(eps, seed):
    """一晚的時間軸：每位住民 SESSIONS 次出現在鏡頭前，每次連做 PER_SESSION 件事；
    他自己的跌倒序列各放進一次出現的最後（跌倒後人就在地上等護理師）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for subj in sorted({e["subject"] for e in eps.values()}):
        normal = [k for k, e in eps.items() if e["subject"] == subj and e["fall"] == 0]
        falls = [k for k, e in eps.items() if e["subject"] == subj and e["fall"] == 1]
        starts = np.sort(rng.choice(np.arange(0, NIGHT_H * 3600 - 300, 300), SESSIONS, replace=False))
        fall_in = dict(zip(rng.choice(SESSIONS, len(falls), replace=False), falls))
        for s, t0 in enumerate(starts):
            chosen = list(rng.choice(normal, PER_SESSION)) + ([fall_in[s]] if s in fall_in else [])
            t = float(t0)
            for k in chosen:
                e = eps[k]
                n = len(e["p"])
                rows.append(pd.DataFrame({"who": subj, "session": f"{subj}-{s}", "t": t + np.arange(n) / FPS,
                                          "p": e["p"], "pose": e["pose"], "fall": e["fall"],
                                          "ep": f"{subj}-{s}-{k}"}))
                t += n / FPS
    return pd.concat(rows).reset_index(drop=True)


def fsm_flags(tl, rearm):
    """每個人一路跑狀態機（離開鏡頭＝新的一次出現就重置）。

    rearm=False：進入 alert 後就停在 alert（每格都是告警）。
    rearm=True ：進入 alert 的那一格發一次告警，然後狀態機重置、重新監看。
    """
    flag = np.zeros(len(tl), bool)
    for _, g in tl.groupby("session", sort=False):
        fsm = FallFSM(threshold=THRESHOLD)
        for i, p, pose in zip(g.index, g["p"], g["pose"]):
            if fsm.step(p, pose) == "alert":
                flag[i] = True
                if rearm:
                    fsm.reset()
    return flag


def notify(tl, flag, merge_s=None, cooldown_s=None):
    """旗標 → 通知。merge_s：與上一個旗標間隔小於它就併成同一件；cooldown_s：同一人通知後多久內不再通知。"""
    out = []
    f = tl[flag]
    for who, g in f.groupby("who"):
        last_flag, last_note = -1e9, -1e9
        for t, fall, ep in zip(g["t"], g["fall"], g["ep"]):
            same = merge_s is not None and t - last_flag < merge_s + 1e-9
            last_flag = t
            if same or (cooldown_s is not None and t - last_note < cooldown_s):
                continue
            last_note = t
            out.append((who, t, fall, ep))
    return pd.DataFrame(out, columns=["who", "t", "fall", "ep"])


def summarize(n, tl):
    falls = tl.loc[tl["fall"] == 1, "ep"].unique()
    hit = set(n.loc[n["fall"] == 1, "ep"])
    return {"通知": len(n), "跌倒相關": int((n["fall"] == 1).sum()), "誤報": int((n["fall"] == 0).sum()),
            "有通知到的跌倒": f"{len(hit)}/{len(falls)}"}


VARIANTS = [
    ("窗口模型直接告警", "每格都通知", lambda tl: tl["p"].to_numpy() >= THRESHOLD, {}),
    ("窗口模型直接告警", "合併", lambda tl: tl["p"].to_numpy() >= THRESHOLD, {"merge_s": MERGE_S}),
    ("窗口模型直接告警", "合併＋冷卻", lambda tl: tl["p"].to_numpy() >= THRESHOLD,
     {"merge_s": MERGE_S, "cooldown_s": COOLDOWN_S}),
    ("狀態機", "停在告警、每格都通知", lambda tl: fsm_flags(tl, rearm=False), {}),
    ("狀態機", "告警一次就重置", lambda tl: fsm_flags(tl, rearm=True), {}),
    ("狀態機", "重置＋冷卻", lambda tl: fsm_flags(tl, rearm=True), {"cooldown_s": COOLDOWN_S}),
]


def main():
    df = prepare(load_poses())
    train, _ = subject_split(df)
    fit, val = train[train["subject_id"].isin(FIT)], train[train["subject_id"].isin(VALID)]
    win, pose = train_models(fit)
    eps = scored_episodes(val, win, pose)

    tl = night(eps, seed=0)
    n_ep = tl.groupby("ep").first()
    print("== 1. 一晚的模擬時間軸（第 0 晚）==")
    print(f"住民 {tl['who'].nunique()} 位（驗證集 S15–S18）× {NIGHT_H} 小時；每人出現 {SESSIONS} 次、每次連做 {PER_SESSION} 件事")
    print(f"鏡頭前的事件 {len(n_ep)} 件，其中跌倒 {int(n_ep['fall'].sum())} 件（每件跌倒後在地上 {ON_FLOOR_S} 秒）")

    print("\n== 2. 同一晚，不同的「通知」算法 ==")
    rows = [{"做法": a, "算法": b, **summarize(notify(tl, fn(tl), **kw), tl)} for a, b, fn, kw in VARIANTS]
    print(pd.DataFrame(rows).set_index(["做法", "算法"]).to_string())

    print("\n== 3. 30 個模擬夜晚：每晚通知次數 ==")
    rows, sources, swallowed = [], {}, 0
    for seed in range(30):
        tl = night(eps, seed)
        for a, b, fn, kw in VARIANTS:
            if b in ("合併＋冷卻", "告警一次就重置", "重置＋冷卻"):
                n = notify(tl, fn(tl), **kw)
                s = summarize(n, tl)
                got, total = map(int, s["有通知到的跌倒"].split("/"))
                rows.append({"做法": f"{a}（{b}）", "每晚通知": s["通知"], "其中誤報": s["誤報"],
                             "漏通知的跌倒": total - got})
                if b == "告警一次就重置":
                    for ep in n.loc[n["fall"] == 0, "ep"]:
                        sess = tl[tl["session"] == tl.loc[tl["ep"] == ep, "session"].iloc[0]]
                        order = list(dict.fromkeys(sess["ep"]))
                        i = order.index(ep)
                        prev = eps[order[i - 1].split("-", 2)[2]]["activity"] if i else "（剛出現）"
                        key = f"{prev} → {eps[ep.split('-', 2)[2]]['activity']}"
                        sources[key] = sources.get(key, 0) + 1
                if b == "重置＋冷卻":
                    swallowed += total - got
    t = pd.DataFrame(rows).groupby("做法", sort=False).agg(["mean", "min", "max"])
    print(t.round(1).to_string())

    print("\n== 4. 狀態機的誤報從哪裡來（30 晚合計，前一件事 → 告警時正在做的事）==")
    for k, v in sorted(sources.items(), key=lambda x: -x[1])[:4]:
        print(f"{k:<22} {v}")
    print(f"冷卻 {COOLDOWN_S:.0f} 秒吃掉的真跌倒：30 晚共 {swallowed} 件（都在同一人誤報後 60 秒內）")


if __name__ == "__main__":
    main()
