"""V29｜健康組 MVP 的元件：骨架串流來源、跌倒推論器（P16 窗口模型＋姿勢模型＋V21 狀態機）、一晚的排程。

V33–V35 也 import 這支：

    import sys; from common import ROOT
    sys.path.insert(0, str(ROOT / "V29-mvp-fall"))
    from fall_pipeline import run_night, build_posture_model, POSTURE_PATH

設計（V29 講解）：
- 取像＋YOLO-pose 這一段，在 mock 模式換成「poses.csv 裡錄好的骨架」：Frame.image 放的是一格骨架（一列），
  不是影像。P18：畫面不離開骨架那一層，事件 API 之後只看得到骨架。
- 窗口特徵：串流只留最近 WINDOW+1 格（V19：max_hip_speed 要 11 格）。
- 跌倒模型：P16 定版 models/fall-hgb.joblib＋門檻（V32 設定檔，抄自 config/fall.yaml）。
- 姿勢模型：P14 的做法（HGB，S01–S18 訓練），定版成 models/posture-hgb.joblib，也有 sha256_12。
- 狀態機：V21 fsm.FallFSM；告警一次就重置（V22）；同一支攝影機冷卻 60 秒（V22、V31 Pipeline）。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # 與 P16 相同：HGB 單執行緒，模型檔可重現
import sys  # noqa: E402
from collections import deque  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import FPS, MODELS, ROOT, WINDOW, WINDOW_FEATURES, frame_features, load_poses, subject_split, window_features  # noqa: E402

for _d in ("V19-pose-stream", "V21-fall-state-machine", "V27-event-api", "V31-mock-tests", "V32-deploy-checklist"):
    sys.path.insert(0, str(ROOT / _d))
from fsm import FallFSM  # noqa: E402
from pipeline import TW, Frame  # noqa: E402

POSTURE_PATH = MODELS / "posture-hgb.joblib"
HOLD_S = 3.0          # 每段序列後「人停在原地」3 秒（V21 hold_last 的同一個假設）
GAP_S = 300           # 同一支攝影機前後兩段相隔 5 分鐘（V22：每 5 分鐘一個時段）
NIGHT_START = datetime(2026, 1, 12, 22, 0, tzinfo=TW)


def build_posture_model(path=POSTURE_PATH):
    """P14 的姿勢分類（四類），用 S01–S18 全部訓練；單執行緒，重跑得到同一個檔。"""
    from sklearn.ensemble import HistGradientBoostingClassifier

    df = load_poses()
    train, _ = subject_split(df)
    m = HistGradientBoostingClassifier(random_state=0).fit(frame_features(train), train["posture"])
    joblib.dump(m, path)
    return m


def load_models(cfg):
    """回傳 (跌倒模型, 姿勢模型)。姿勢模型不在就建一次。"""
    fall = joblib.load(ROOT / cfg.model.path)
    pose = joblib.load(POSTURE_PATH) if POSTURE_PATH.exists() else build_posture_model()
    return fall, pose


def with_hold(seq: pd.DataFrame, seconds=HOLD_S) -> pd.DataFrame:
    """序列最後一格再重複 seconds 秒（模擬人還停在原地；poses.csv 每段只有 4 秒）。"""
    seq = seq.sort_values("frame")
    last = seq.tail(1)
    n = int(seconds * FPS)
    extra = pd.concat([last] * n, ignore_index=True)
    extra["frame"] = int(last["frame"].iloc[0]) + 1 + np.arange(n)
    return pd.concat([seq, extra], ignore_index=True)


def schedule(df: pd.DataFrame, seed=0) -> pd.DataFrame:
    """一支攝影機一晚的播放順序：該攝影機的序列亂序（rng seed），每段開始時間相隔 GAP_S 秒。

    回傳的每一列是一格骨架，多 t_wall（帶時區的時間）與 slot（第幾段）欄。
    """
    rows = []
    for cam, g in df.groupby("camera_id"):
        ids = sorted(g["seq_id"].unique())
        order = np.random.default_rng(seed).permutation(ids)
        for slot, sid in enumerate(order):
            s = with_hold(g[g["seq_id"] == sid])
            t0 = NIGHT_START + timedelta(seconds=slot * GAP_S)
            s["t_wall"] = [t0 + timedelta(seconds=f / FPS) for f in range(len(s))]
            s["slot"] = slot
            rows.append(s)
    return pd.concat(rows, ignore_index=True)


class SkeletonSource:
    """V31 的取像介面：grab() → Frame。image 欄放「一格骨架」（DataFrame 一列），不是影像。"""

    def __init__(self, frames: pd.DataFrame):
        self.frames = frames.reset_index(drop=True)
        self.i = 0

    def grab(self) -> Frame:
        row = self.frames.iloc[[self.i]]
        self.i += 1
        return Frame(row, row["t_wall"].iloc[0])

    def __len__(self):
        return len(self.frames)


class FallDetector:
    """V31 的推論器介面：predict(一格骨架) → {"score", "evidence"}。有狀態：窗口緩衝＋V21 狀態機。

    score：狀態機進入 alert 的那一格＝這次「疑似」期間窗口模型的最高機率（一定 ≥ 門檻）；其他格 0。
    告警一次就重置狀態機（V22）。換一段序列（新的人進入畫面）也重置。
    """

    def __init__(self, fall_model, pose_model, threshold):
        self.fall, self.pose, self.threshold = fall_model, pose_model, threshold
        self.fsm = FallFSM(threshold=threshold)
        self.buf: deque = deque(maxlen=WINDOW + 1)
        self.seq = None
        self.peak = 0.0
        self.trace: list[dict] = []

    def predict(self, row: pd.DataFrame) -> dict:
        assert isinstance(row, pd.DataFrame) and len(row) == 1, "健康組的 Frame.image 是一格骨架"
        sid = row["seq_id"].iloc[0]
        if sid != self.seq:
            self.seq, self.peak = sid, 0.0
            self.fsm.reset()
            self.buf.clear()
        if "p_fall" in row and "pose_pred" in row:          # 已用 score_batch 算好（與串流逐格相同，見 mvp_fall 第 2 段）
            p, g = float(row["p_fall"].iloc[0]), str(row["pose_pred"].iloc[0])
        else:                                                # 真的串流：只用最近 11 格
            self.buf.append(row)
            w = pd.concat(list(self.buf), ignore_index=True)
            p = float(self.fall.predict_proba(window_features(w)[WINDOW_FEATURES].tail(1))[0, 1])
            g = str(self.pose.predict(frame_features(row))[0])
        before = self.fsm.state
        st = self.fsm.step(p, g)
        if st == "suspected" and before == "standing":
            self.peak = p
        elif st == "suspected":
            self.peak = max(self.peak, p)
        score = 0.0
        if st == "alert":
            score = max(self.peak, p)
            self.fsm.reset()
        self.trace.append({"seq_id": sid, "frame": int(row["frame"].iloc[0]), "p": p, "pose": g, "state": st})
        # 注意：evidence 不能用 "frame" 當鍵——V27 把它視為影像欄位而拒收（422）
        return {"score": round(score, 3), "evidence": {"seq_id": sid, "frame_no": int(row["frame"].iloc[0]),
                                                       "keypoints_only": True}}


def score_batch(frames, fall_model, pose_model):
    """整批算每格的 p_fall、pose_pred（每段序列各自算窗口，不跨序列）。結果與串流逐格相同，只是快很多。"""
    out = frames.copy()
    out["p_fall"] = fall_model.predict_proba(window_features(out)[WINDOW_FEATURES])[:, 1]
    out["pose_pred"] = pose_model.predict(frame_features(out))
    return out


def run_camera(frames, cfg, cam, fall_model, pose_model, sender):
    """一支攝影機跑完一晚。回傳 (Pipeline, FallDetector)。"""
    from pipeline import PipeConfig, Pipeline

    pc = PipeConfig(source="pose", site=cfg.site, camera_id=cam.id, prefix=cam.prefix,
                    threshold=cfg.threshold.value,
                    model={"name": cfg.model.name, "version": cfg.model.version, "sha256_12": cfg.model.sha256_12},
                    cooldown_s=cfg.cooldown_s)
    det = FallDetector(fall_model, pose_model, cfg.threshold.value)
    pipe = Pipeline(SkeletonSource(frames), det, sender, pc)
    for _ in range(len(frames)):
        pipe.step()
    return pipe, det


def run_night(df, cfg, client, seed=0, stream=False):
    """全部攝影機跑一晚，事件送到 client（V27）。回傳 (每段序列的結果表, {cam: Pipeline})。"""
    from pipeline import ApiSender

    fall_model, pose_model = load_models(cfg)
    plan = schedule(df, seed)
    if not stream:
        plan = score_batch(plan, fall_model, pose_model)
    sender = ApiSender(client)
    pipes = {}
    for cam in cfg.cameras:
        frames = plan[plan["camera_id"] == cam.id]
        if len(frames):
            pipes[cam.id] = run_camera(frames, cfg, cam, fall_model, pose_model, sender)[0]
    events = client.get("/events", params={"kind": "fall"}).json()
    first = {}
    for e in events:
        sid = e["evidence"]["seq_id"]
        first.setdefault(sid, e)
    seq = plan.groupby("seq_id").agg(camera_id=("camera_id", "first"), activity=("activity", "first"),
                                     y=("is_fall_seq", "first"), fall_frame=("fall_frame", "first"),
                                     start=("t_wall", "first"))
    seq["alarm"] = seq.index.isin(first.keys()).astype(int)
    seq["alert_frame"] = [first[s]["evidence"]["frame_no"] if s in first else -1 for s in seq.index]
    seq["score"] = [first[s]["score"] if s in first else np.nan for s in seq.index]
    return seq, pipes, events
