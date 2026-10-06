"""V27｜事件 JSON 與事件 API（兩組共用的告警後端）。

V28–V31 都 import 這個模組，不要各自重寫：

    import sys; sys.path.insert(0, "V27-event-api")      # 從 demo/ 執行時
    from event_api import Event, create_app, make_client, make_event_id, EXAMPLE_FALL, EXAMPLE_NG

    client = make_client(create_app())                      # TestClient；預設 SQLite ":memory:"
    r = client.post("/events", json=EXAMPLE_FALL)           # 201 {"id": ...}
    client.get("/events", params={"kind": "fall", "since": "2026-01-12T00:00:00+08:00"})
    client.post(f"/events/{r.json()['id']}/ack", json={"by": "N02", "verdict": "false_alarm"})

設計決定（V27 講解）：
- 事件只描述「發生了什麼、模型多有把握、用哪一版模型」，**不存影像本體**（P18 隱私決策）。
- 時間一律帶時區；資料庫另存 UTC 秒數，查詢 since 用它比較。
- ack 只能做一次：第一位到場的人說了算，要更正請另開紀錄（V30）。
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

KINDS = {"pose": "fall", "defect": "ng"}          # source → 允許的 kind
VERDICTS = ("true_alarm", "false_alarm")
IMAGE_KEYS = {"image", "img", "frame", "jpeg", "png", "base64", "image_b64", "photo"}
MAX_EVIDENCE_STR = 200                              # evidence 裡的字串超過這個長度，多半是偷塞影像


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sha256_12: str = Field(pattern=r"^[0-9a-f]{12}$")


class Event(BaseModel):
    """一筆告警事件。兩組共用；欄位定義見 BRIEF「跨節介面」。"""
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(pattern=r"^[A-Za-z0-9_-]{3,64}$")
    source: Literal["pose", "defect"]
    site: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    event_time: datetime
    kind: Literal["fall", "ng"]
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    model: ModelInfo

    @field_validator("event_time")
    @classmethod
    def _need_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("event_time 必須帶時區，例如 +08:00")
        return v

    @field_validator("evidence")
    @classmethod
    def _no_image(cls, v: dict) -> dict:
        for k, x in v.items():
            if k.lower() in IMAGE_KEYS:
                raise ValueError(f"evidence 不可放影像本體（欄位 {k}）")
            if isinstance(x, str) and len(x) > MAX_EVIDENCE_STR:
                raise ValueError(f"evidence.{k} 太長（{len(x)} 字元），疑似影像或 base64")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "Event":
        if KINDS[self.source] != self.kind:
            raise ValueError(f"source={self.source} 的 kind 應為 {KINDS[self.source]}，收到 {self.kind}")
        if self.score < self.threshold:
            raise ValueError(f"score {self.score} 低於 threshold {self.threshold}：未達門檻不該送出事件")
        return self


class Ack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    by: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    verdict: Literal["true_alarm", "false_alarm"]


def make_event_id(prefix: str, event_time: datetime, n: int = 0) -> str:
    """'s1cam2' + 2026-01-12T02:15:03+08:00 → 's1cam2-20260112-021503-0'（同一秒第 n 筆）。"""
    return f"{prefix}-{event_time:%Y%m%d-%H%M%S}-{n}"


def _utc(dt: datetime) -> float:
    return dt.astimezone(timezone.utc).timestamp()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EventStore:
    """SQLite 儲存。db_path=":memory:" 時只活在這個物件裡（測試用）。"""

    def __init__(self, db_path: str = ":memory:", now: Callable[[], datetime] = _now):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.now = now
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY, kind TEXT NOT NULL, site TEXT NOT NULL, camera_id TEXT NOT NULL,
            t_utc REAL NOT NULL, body TEXT NOT NULL,
            ack_by TEXT, ack_verdict TEXT, ack_at TEXT)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_kind_t ON events(kind, t_utc)")
        self.db.commit()

    def add(self, ev: Event) -> bool:
        with self.lock:
            try:
                self.db.execute("INSERT INTO events(event_id, kind, site, camera_id, t_utc, body) VALUES (?,?,?,?,?,?)",
                                (ev.event_id, ev.kind, ev.site, ev.camera_id, _utc(ev.event_time),
                                 ev.model_dump_json()))
            except sqlite3.IntegrityError:
                return False
            self.db.commit()
            return True

    def list(self, kind: str | None = None, since: datetime | None = None) -> list[dict]:
        sql, args = "SELECT body, ack_by, ack_verdict, ack_at FROM events WHERE 1=1", []
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if since:
            sql += " AND t_utc >= ?"
            args.append(_utc(since))
        with self.lock:
            rows = self.db.execute(sql + " ORDER BY t_utc, event_id", args).fetchall()
        out = []
        for body, by, verdict, at in rows:
            d = json.loads(body)
            d["ack"] = None if by is None else {"by": by, "verdict": verdict, "at": at}
            out.append(d)
        return out

    def ack(self, event_id: str, a: Ack) -> str:
        """回傳 'ok'、'missing' 或 'already'。"""
        with self.lock:
            row = self.db.execute("SELECT ack_by FROM events WHERE event_id = ?", (event_id,)).fetchone()
            if row is None:
                return "missing"
            if row[0] is not None:
                return "already"
            self.db.execute("UPDATE events SET ack_by=?, ack_verdict=?, ack_at=? WHERE event_id=?",
                            (a.by, a.verdict, self.now().isoformat(timespec="seconds"), event_id))
            self.db.commit()
            return "ok"

    def count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]


def create_app(db_path: str = ":memory:", now: Callable[[], datetime] = _now) -> FastAPI:
    """建立事件 API。測試用 TestClient(create_app())；上線用 uvicorn 載入 create_app("events.db")。

    now：ack 時間的來源（測試或模擬時可傳固定時鐘，讓輸出可重現）。
    """
    app = FastAPI(title="看見科技 事件 API", version="2026.01")
    store = EventStore(db_path, now)
    app.state.store = store

    @app.post("/events", status_code=201)
    def post_event(ev: Event):
        if not store.add(ev):
            raise HTTPException(409, f"event_id 重複：{ev.event_id}")
        return {"id": ev.event_id}

    @app.get("/events")
    def get_events(kind: Literal["fall", "ng"] | None = None, since: datetime | None = Query(None)):
        if since is not None and since.tzinfo is None:
            raise HTTPException(422, "since 必須帶時區，例如 2026-01-12T00:00:00+08:00")
        return store.list(kind, since)

    @app.post("/events/{event_id}/ack")
    def ack_event(event_id: str, a: Ack):
        r = store.ack(event_id, a)
        if r == "missing":
            raise HTTPException(404, f"找不到事件：{event_id}")
        if r == "already":
            raise HTTPException(409, f"事件已確認過：{event_id}")
        return {"id": event_id, "verdict": a.verdict}

    @app.get("/health")
    def health():
        return {"ok": True, "events": store.count()}

    return app


def make_client(app: FastAPI | None = None):
    """回傳 TestClient（不起真的 server）；順便關掉 starlette 對 httpx 的棄用警告，讓輸出乾淨。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    return TestClient(app if app is not None else create_app())


# ---------------------------------------------------------------- 兩組的範例事件（V27 講義、V28–V31 測試共用）

EXAMPLE_FALL = {
    "event_id": "s1cam2-20260112-021503-0",
    "source": "pose",
    "site": "松年本館",
    "camera_id": "cam2",
    "event_time": "2026-01-12T02:15:03+08:00",
    "kind": "fall",
    "score": 0.98,
    "threshold": 0.95,
    "evidence": {"seq_id": "SQ0123", "keypoints_only": True},
    "model": {"name": "fall-hgb", "version": "2026.01", "sha256_12": "05ba1c2d31cf"},
}

EXAMPLE_NG = {
    "event_id": "line1cam1-20260112-101422-0",
    "source": "defect",
    "site": "精工五金",
    "camera_id": "line1cam1",
    "event_time": "2026-01-12T10:14:22+08:00",
    "kind": "ng",
    "score": 0.87,
    "threshold": 0.62,
    "evidence": {"part_id": "P0207", "category": "screw", "heatmap_max_xy": [131, 88],
                 "thumb": "thumbs/P0207.jpg"},
    "model": {"name": "patch-knn-screw", "version": "2026.01", "sha256_12": "a41d6e90c3b8"},
}
