"""V23：同一個模型、同一個門檻，搬到二館。先看沒有標籤也能看的指標，再用 30% 有標籤的序列量 recall。

模型：S01–S18 全部本館訓練受試者重新訓練（上線版）。本館測試集 S19–S24 在這裡第二次打開，
只當「本館的參考值」，不拿來選任何東西。

    python V23-site-drift/drift.py
"""
import sys

import numpy as np
import pandas as pd

from common import DATA, ROOT, WINDOW_FEATURES, load_poses, subject_split, window_features

sys.path.insert(0, str(ROOT / "V19-pose-stream"))
sys.path.insert(0, str(ROOT / "V21-fall-state-machine"))
from fallmodel import prepare, score_frames, train_models  # noqa: E402
from fsm import hold_last, run_seq  # noqa: E402

THRESHOLD = 0.5
SHOW = ["torso_angle", "aspect", "hip_height", "knee_angle", "mean_conf", "max_hip_speed"]


def load_site2():
    d = load_poses(DATA / "pose/poses-site2.csv")
    d["labeled"] = d["activity"] != ""
    return prepare(d)


def seq_results(d, win, pose, threshold=THRESHOLD):
    """逐格分數 → 每段序列：窗口最高分、狀態機有沒有告警、標籤（沒有標籤＝NaN）。"""
    v = score_frames(prepare(hold_last(d, 3.0)), win, pose)
    s = v.groupby("seq_id").agg(camera=("camera_id", "first"), subject=("subject_id", "first"),
                                score=("p_fall", "max"), y=("is_fall_seq", "first"))
    s["y"] = pd.to_numeric(s["y"], errors="coerce")
    s["alert"] = pd.Series({sid: "alert" in run_seq(g, threshold=threshold) for sid, g in v.groupby("seq_id")})
    return v, s


def main():
    df = prepare(load_poses())
    train, test = subject_split(df)
    s2 = load_site2()
    win, pose = train_models(train)

    print("== 1. 二館的資料 ==")
    seq2 = s2.groupby("seq_id").agg(cam=("camera_id", "first"), lab=("labeled", "first"), y=("is_fall_seq", "first"))
    print(f"序列 {len(seq2)} 段（T01–T10，攝影機 {', '.join(sorted(seq2['cam'].unique()))}）；"
          f"有標籤 {int(seq2['lab'].sum())} 段（{seq2['lab'].mean():.0%}），其中跌倒 {int((seq2['y'] == '1').sum())} 段")

    print("\n== 2. 不用標籤：特徵分布（中位數）==")
    F = {"本館訓練 S01–S18": window_features(train), "本館測試 S19–S24": window_features(test),
         "二館（全部）": window_features(s2)}
    t = pd.DataFrame({k: f[SHOW].median() for k, f in F.items()}).T
    ref = F["本館訓練 S01–S18"]
    t.loc["二館偏移（以本館標準差計）"] = (F["二館（全部）"][SHOW].median() - ref[SHOW].median()) / ref[SHOW].std()
    print(t.round(3).to_string())

    print("\n== 3. 不用標籤：上線後每天就能看的指標 ==")
    rows = {}
    for name, d in [("本館測試 S19–S24", test), ("二館（全部）", s2)]:
        v, s = seq_results(d, win, pose)
        f = F["本館測試 S19–S24" if name.startswith("本館") else "二館（全部）"]
        conf = v[[c for c in v.columns if c.endswith("_c")]].to_numpy()
        rows[name] = {"狀態機告警率": s["alert"].mean(), "窗口分數≥門檻的序列": (s["score"] >= THRESHOLD).mean(),
                      "窗口分數 0.2–0.8 的序列": s["score"].between(0.2, 0.8).mean(),
                      "關鍵點信心 <0.3 比例": (conf < 0.3).mean(),
                      "特徵 NaN 比例": f[WINDOW_FEATURES].isna().to_numpy().mean(),
                      "姿勢模型判 bending 比例": (v["pose_pred"] == "bending").mean()}
    print(pd.DataFrame(rows).round(3).to_string())

    print("\n== 4. 有標籤的 30%：實際表現 ==")
    rows = {}
    for name, d in [("本館測試 S19–S24", test), ("二館有標籤", s2[s2["labeled"]])]:
        v, s = seq_results(d, win, pose)
        fall, ok = s[s["y"] == 1], s[s["y"] == 0]
        miss, fa = int((~fall["alert"]).sum()), int(ok["alert"].sum())
        lab = v[v["posture"] != ""]
        rows[name] = {"序列": len(s), "跌倒": len(fall), "狀態機 recall": round(fall["alert"].mean(), 3),
                      "漏報": miss, "誤報": fa, "成本": 10 * miss + fa,
                      "窗口直接告警 recall": round((fall["score"] >= THRESHOLD).mean(), 3),
                      "窗口直接告警 誤報": int((ok["score"] >= THRESHOLD).sum()),
                      "姿勢模型準確率": round((lab["pose_pred"] == lab["posture"]).mean(), 3)}
    print(pd.DataFrame.from_dict(rows, orient="index").to_string())
    _, s = seq_results(s2[s2["labeled"]], win, pose)
    print("二館跌倒序列的窗口最高分：", " ".join(f"{x:.2f}" for x in sorted(s.loc[s["y"] == 1, "score"])))


if __name__ == "__main__":
    main()
