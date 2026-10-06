"""V20：把兩段不同受試者的序列合併成「一個畫面兩個人」，每格順序打亂，再追蹤。

    python V20-tracking/track.py
"""
import warnings

import numpy as np
import pandas as pd
import sys

from common import ROOT, keypoints, load_poses, subject_split, window_features

sys.path.insert(0, str(ROOT / "V19-pose-stream"))
from fallmodel import FIT, prepare, train_models  # noqa: E402

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # 沒有 scipy 就退回貪婪法
    linear_sum_assignment = None

warnings.simplefilter("ignore", RuntimeWarning)
VALID = [f"S{i:02d}" for i in range(15, 19)]


def boxes(rows: pd.DataFrame) -> np.ndarray:
    """骨架 → 外框 (x1, y1, x2, y2)，只用信心夠的點。"""
    k = keypoints(rows)
    return np.stack([np.nanmin(k[..., 0], 1), np.nanmin(k[..., 1], 1),
                     np.nanmax(k[..., 0], 1), np.nanmax(k[..., 1], 1)], 1)


def make_scene(df, seq_a, seq_b, seed):
    """兩段序列疊在同一個畫面：每格兩副骨架，順序隨機（模型不會照人排序）。"""
    rng = np.random.default_rng(seed)
    a = df[df["seq_id"] == seq_a].sort_values("frame")
    b = df[df["seq_id"] == seq_b].sort_values("frame")
    rows = []
    for f in range(40):
        pair = [("A", a.iloc[[f]]), ("B", b.iloc[[f]])]
        if rng.random() < 0.5:
            pair.reverse()
        for who, r in pair:
            rows.append({"frame": f, "truth": who, "row": r.index[0], "box": boxes(r)[0]})
    return pd.DataFrame(rows)


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
    return inter / union if union > 0 else 0.0


def center_dist(p, q):
    return float(np.hypot((p[0] + p[2] - q[0] - q[2]) / 2, (p[1] + p[3] - q[1] - q[3]) / 2))


def cost_matrix(prev, cur, kind):
    if kind == "iou":
        return np.array([[1 - iou(p, c) for c in cur] for p in prev])
    return np.array([[center_dist(p, c) for c in cur] for p in prev])


def greedy(cost):
    """貪婪法：每次挑整張表裡最小的一格，劃掉那一列與那一欄。"""
    cost = cost.astype(float).copy()
    rows, cols = [], []
    for _ in range(min(cost.shape)):
        r, c = np.unravel_index(np.argmin(cost), cost.shape)
        rows.append(r)
        cols.append(c)
        cost[r, :] = np.inf
        cost[:, c] = np.inf
    return np.array(rows), np.array(cols)


def track(scene, kind="iou", method="hungarian"):
    """回傳每列偵測被指派的 track id。kind=none 表示不追蹤：直接用偵測順序當 ID。"""
    ids = np.zeros(len(scene), int)
    last = None  # 上一格每個 track 的外框
    for f, g in scene.groupby("frame", sort=True):
        cur = list(g["box"])
        if kind == "none" or last is None:
            assign = list(range(len(cur)))
        else:
            cost = cost_matrix(last, cur, kind)
            solver = linear_sum_assignment if (method == "hungarian" and linear_sum_assignment) else greedy
            r, c = solver(cost)
            assign = [0] * len(cur)
            for ti, di in zip(r, c):
                assign[di] = ti
        ids[g.index] = assign
        last = [None] * len(cur)
        for di, ti in enumerate(assign):
            last[ti] = cur[di]
    return ids


def id_switches(scene, ids):
    """同一個真實的人，track id 和上一格不同就算一次 ID switch。"""
    s = scene.assign(tid=ids).sort_values("frame")
    return int(sum((g["tid"].diff().fillna(0) != 0).sum() for _, g in s.groupby("truth")))


def scenes(df):
    """驗證集受試者兩兩配對：走路對走路（會交錯）、走路對跌倒、坐下對躺床。"""
    seq = df[df["subject_id"].isin(VALID)].groupby("seq_id").first()
    out = []
    pairs = [("walk", "walk"), ("walk", "fall"), ("sit_down", "lie_bed"), ("bend_pick", "fall")]
    for i, (x, y) in enumerate(pairs):
        xs = seq[seq["activity"] == x]
        ys = seq[seq["activity"] == y]
        for j, (sa, ra) in enumerate(xs.iterrows()):
            cand = ys[(ys["subject_id"] != ra["subject_id"]) & (ys["camera_id"] == ra["camera_id"])]
            if len(cand):
                sb = cand.index[j % len(cand)]
                if sa != sb:
                    out.append((f"{x}+{y}", sa, sb, 100 * i + j))
    return out


def main():
    df = load_poses()
    print("== 1. 一個畫面兩個人：偵測器吐出來的順序是亂的 ==")
    sc = make_scene(df, *scenes(df)[0][1:3], seed=0)
    for f in range(4):
        g = sc[sc["frame"] == f]
        print(f"frame {f}: " + "  ".join(f"[{i}] 真={t} 中心x={(b[0] + b[2]) / 2:5.1f}"
                                         for i, (t, b) in enumerate(zip(g["truth"], g["box"]))))

    print("\n== 2. 四種做法的 ID switch（驗證集 S15–S18 配對場景）==")
    rows = []
    all_sc = [(name, make_scene(df, a, b, seed)) for name, a, b, seed in scenes(df)]
    methods = [("不追蹤（照偵測順序）", "none", "-"), ("中心距離＋貪婪", "center", "greedy"),
               ("中心距離＋匈牙利", "center", "hungarian"), ("IoU＋匈牙利", "iou", "hungarian")]
    for label, kind, method in methods:
        per = {}
        for name, s in all_sc:
            per.setdefault(name, 0)
            per[name] += id_switches(s, track(s, kind, method))
        rows.append({"做法": label, **per, "合計": sum(per.values())})
    t = pd.DataFrame(rows).set_index("做法")
    print(t.to_string())
    n = pd.Series([name for name, _ in all_sc]).value_counts()
    print("場景數：", n.reindex(t.columns[:-1]).to_dict(), "；每個場景 40 格 × 2 人")
    print("\n== 3. 追蹤錯了，下游的跌倒模型會看到什麼 ==")
    train, _ = subject_split(prepare(df))
    win, _ = train_models(train[train["subject_id"].isin(FIT)])
    rows = []
    for label, kind, method in [methods[0], methods[3]]:
        alerts = {"有跌倒的場景": [0, 0], "沒有跌倒的場景": [0, 0]}
        for name, s in all_sc:
            ids = track(s, kind, method)
            seqs = []
            for tid in (0, 1):
                r = df.loc[s.loc[ids == tid, "row"]].copy()
                r["frame"] = s.loc[ids == tid, "frame"].to_numpy()
                r["seq_id"] = f"track{tid}"
                seqs.append(r)
            d = pd.concat(seqs).sort_values(["seq_id", "frame"]).reset_index(drop=True)
            fired = (win.predict_proba(window_features(d))[:, 1] >= 0.5).any()
            key = "有跌倒的場景" if "fall" in name else "沒有跌倒的場景"
            alerts[key][0] += int(fired)
            alerts[key][1] += 1
        rows.append({"做法": label, **{k: f"{a}/{n}" for k, (a, n) in alerts.items()}})
    print(pd.DataFrame(rows).set_index("做法").to_string())
    print("（數字＝窗口模型機率 ≥ 0.5 的場景數／場景總數）")
    print("\nscipy：", "有（匈牙利法）" if linear_sum_assignment else "沒有（退回貪婪法）")


if __name__ == "__main__":
    main()
