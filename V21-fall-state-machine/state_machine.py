"""V21：規則與模型合作。比較「單格模型直接告警」「窗口模型直接告警」「只用姿勢規則」與「狀態機」。

訓練 S01–S14、驗證 S15–S18（以序列為單位）；測試集 S19–S24 不打開。

    python V21-fall-state-machine/state_machine.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from common import FPS, ROOT, frame_features, load_poses, subject_split

sys.path.insert(0, str(ROOT / "V19-pose-stream"))
sys.path.insert(0, str(ROOT / "V21-fall-state-machine"))
from fallmodel import FIT, GROUND, VALID, cost, prepare, score_frames, train_models  # noqa: E402
from fsm import hold_last, run_seq  # noqa: E402

THRESHOLD = 0.5
CODE = {"standing": "S", "suspected": "?", "alert": "!"}
POSE = {"standing": "S", "sitting": "s", "lying": "L", "bending": "b"}


def ground_rule(d, need=2 * FPS):
    """只用姿勢規則：姿勢模型連續 need 格躺／坐地就告警。"""
    run = 0
    for g in d["pose_pred"]:
        run = run + 1 if g in GROUND else 0
        if run >= need:
            return True
    return False


def evaluate(v, frame_model):
    v = v.copy()
    v["p_frame"] = frame_model.predict_proba(frame_features(v))[:, 1]
    rows, first = {}, {}
    for sid, d in v.groupby("seq_id"):
        st = run_seq(d, threshold=THRESHOLD)
        first[sid] = st.index("alert") if "alert" in st else -1
        rows[sid] = {"單格模型直接告警": (d["p_frame"] >= THRESHOLD).any(),
                     "窗口模型直接告警": (d["p_fall"] >= THRESHOLD).any(),
                     "姿勢規則（躺坐 2 秒）": ground_rule(d),
                     "狀態機": "alert" in st,
                     "_end": st[-1]}
    r = pd.DataFrame(rows).T
    meta = v.groupby("seq_id").agg(y=("is_fall_seq", "first"), activity=("activity", "first"),
                                   ff=("fall_frame", "first"))
    return r.join(meta), pd.Series(first)


def main():
    df = prepare(load_poses())
    train, _ = subject_split(df)
    fit, val = train[train["subject_id"].isin(FIT)], train[train["subject_id"].isin(VALID)]
    win, pose = train_models(fit)
    frame_model = HistGradientBoostingClassifier(random_state=0).fit(frame_features(fit), fit["y_win"])

    v = score_frames(val, win, pose)
    sid = "SQ0221"
    d = v[v["seq_id"] == sid]
    st = run_seq(d, threshold=THRESHOLD)
    print(f"== 1. 一段跌坐（{sid}，fall_frame={int(d['fall_frame'].iloc[0])}）逐格走一次狀態機 ==")
    print("姿勢模型：", "".join(POSE[x] for x in d["pose_pred"]))
    print("窗口機率：", "".join(str(min(9, int(p * 10))) for p in d["p_fall"]), "（0–9＝機率的十分位）")
    print("狀態    ：", "".join(CODE[x] for x in st), "（S 站立、? 疑似、! 告警）")
    print(f"第 {st.index('suspected')} 格進入疑似、第 {st.index('alert')} 格告警")

    print(f"\n== 2. 驗證集 S15–S18（{val['seq_id'].nunique()} 段），門檻 {THRESHOLD}：原始 4 秒片段 ==")
    r, _ = evaluate(v, frame_model)
    methods = ["單格模型直接告警", "窗口模型直接告警", "姿勢規則（躺坐 2 秒）", "狀態機"]

    def table(r):
        out = []
        for m in methods:
            miss = int(((r["y"] == 1) & ~r[m].astype(bool)).sum())
            fa_rows = r[(r["y"] == 0) & r[m].astype(bool)]
            out.append({"做法": m, "漏報": miss, "誤報": len(fa_rows), "成本": cost(miss, len(fa_rows)),
                        "誤報來自": ",".join(f"{k}×{n}" for k, n in fa_rows["activity"].value_counts().items()) or "-"})
        return pd.DataFrame(out).set_index("做法")
    print(table(r).to_string())
    left = r[(r["y"] == 1) & (r["_end"] == "suspected")]
    print(f"狀態機的漏報：{len(left)} 段跌倒在片段結束時還停在「疑似」（片段只有 4 秒）")

    print("\n== 3. 片段結束後保持最後姿勢 3 秒（人還在原地）再評估 ==")
    vh = score_frames(prepare(hold_last(val, 3.0)), win, pose)
    rh, first = evaluate(vh, frame_model)
    print(table(rh).to_string())
    falls = rh[rh["y"] == 1]
    lat = (first[falls.index] - falls["ff"].astype(int)) / FPS
    direct = {sid: int(np.argmax(d["p_fall"].to_numpy() >= THRESHOLD)) for sid, d in vh.groupby("seq_id")
              if sid in falls.index}
    lat_d = (pd.Series(direct) - falls["ff"].astype(int)) / FPS
    print(f"窗口模型直接告警從開始跌倒到告警：{lat_d.min():.1f}～{lat_d.max():.1f} 秒，中位數 {lat_d.median():.1f} 秒")
    print(f"狀態機從開始跌倒到告警：{lat.min():.1f}～{lat.max():.1f} 秒，中位數 {lat.median():.1f} 秒")


if __name__ == "__main__":
    main()
