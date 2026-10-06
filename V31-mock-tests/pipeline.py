"""V31｜把管線拆成三段：取像（Source）→ 推論（Detector）→ 送 API（Sender）。每一段都可以換成假的。

    from pipeline import Pipeline, make_source, MockCamera, FakeDetector, ApiSender

    source = make_source()                      # NTUNHS_MOCK=1 → MockCamera（讀 ezcam-sample.jpg，不連網路）
    pipe = Pipeline(source, FakeDetector([0.1, 0.9]), ApiSender(client), cfg)
    pipe.step()                                 # 取一張 → 推論 → 過門檻就送事件

三段之間只靠很小的介面溝通：
- Source.grab() -> Frame(image: np.ndarray HxWx3 uint8, t: datetime 帶時區)
- Detector.predict(image) -> {"score": float, "evidence": dict}
- Sender.send(event: dict) -> "sent" | "duplicate" | "rejected" | "queued"
V28／V29 的真推論器只要實作 predict()，就能直接放進來。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

import cv2
import numpy as np

from common import PHOTOS

TW = timezone(timedelta(hours=8))


def is_mock() -> bool:
    return os.environ.get("NTUNHS_MOCK") == "1"


# ---------------------------------------------------------------- 1. 取像


@dataclass
class Frame:
    image: np.ndarray
    t: datetime


class Source(Protocol):
    def grab(self) -> Frame: ...


class MockCamera:
    """Mock 模式：每次都回傳 EZCam 真實拍過的那一張 320×240 照片，時間由 clock 決定。"""

    def __init__(self, clock: Callable[[], datetime] = lambda: datetime.now(TW),
                 path=PHOTOS / "ezcam-sample.jpg"):
        self.image = cv2.imread(str(path))
        if self.image is None:
            raise FileNotFoundError(path)
        self.clock = clock

    def grab(self) -> Frame:
        return Frame(self.image.copy(), self.clock())


class EZCamSource:
    """真的 EZCam：要求拍照 → 輪詢最新影像 → 下載 JPEG。金鑰只從環境變數 EZCAM_API_KEY 讀。

    EZCam 是「輪詢拍照」（約 3 秒一張），不是串流；見 MCP_SAMPLE.md。
    """

    def __init__(self, base_url: str = "https://pi5-cam.4impact.cc", wait_s: float = 15.0):
        key = os.environ.get("EZCAM_API_KEY")
        if not key:
            raise RuntimeError("沒有設定 EZCAM_API_KEY；測試或課堂請設 NTUNHS_MOCK=1")
        self.base, self.headers, self.wait_s = base_url.rstrip("/"), {"X-API-Key": key}, wait_s

    def grab(self) -> Frame:  # pragma: no cover（課堂不連網路）
        import time

        import httpx
        with httpx.Client(timeout=10, headers=self.headers) as c:
            before = c.get(f"{self.base}/api/latest").json().get("ts")
            c.post(f"{self.base}/api/capture").raise_for_status()
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.wait_s:
                info = c.get(f"{self.base}/api/latest").json()
                if info.get("has") and info.get("ts") != before:
                    jpg = c.get(info["url"]).content
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    return Frame(img, datetime.now(TW))
                time.sleep(1.0)
        raise TimeoutError(f"EZCam {self.wait_s:.0f} 秒內沒有新照片")


def make_source(clock: Callable[[], datetime] = lambda: datetime.now(TW)) -> Source:
    """NTUNHS_MOCK=1 一律用 MockCamera；否則才去建 EZCamSource（需要 EZCAM_API_KEY）。"""
    return MockCamera(clock) if is_mock() else EZCamSource()


# ---------------------------------------------------------------- 2. 推論


class Detector(Protocol):
    def predict(self, image: np.ndarray) -> dict: ...


class FakeDetector:
    """假推論器：照劇本依序回傳分數，不看影像內容。用來測「管線的其他部分」對不對。"""

    def __init__(self, scores, evidence: dict | None = None):
        self.scores = list(scores)
        self.evidence = evidence or {}
        self.calls = 0

    def predict(self, image: np.ndarray) -> dict:
        assert image.ndim == 3 and image.dtype == np.uint8, "推論器只收 HxWx3 uint8"
        s = self.scores[self.calls % len(self.scores)]
        self.calls += 1
        return {"score": s, "evidence": dict(self.evidence)}


# ---------------------------------------------------------------- 3. 送 API


class ApiSender:
    """把事件 POST 到事件 API。client 可以是 TestClient（測試）或 httpx.Client(base_url=...)（上線）。

    201 → sent；409 → duplicate（之前送到過，當成成功）；422 → rejected（資料有錯，不重送）；
    連線錯誤或 5xx → queued（留在佇列，下一次 flush 再送）。
    """

    def __init__(self, client):
        self.client = client
        self.queue: list[dict] = []
        self.rejected: list[tuple[str, str]] = []

    def _post(self, ev: dict) -> str:
        try:
            r = self.client.post("/events", json=ev)
        except Exception:
            return "queued"
        if r.status_code == 201:
            return "sent"
        if r.status_code == 409:
            return "duplicate"
        if r.status_code == 422:
            self.rejected.append((ev.get("event_id", "?"), str(r.json()["detail"][0]["loc"][-1])))
            return "rejected"
        return "queued"

    def send(self, ev: dict) -> str:
        self.flush()
        status = self._post(ev) if not self.queue else "queued"
        if status == "queued":
            self.queue.append(ev)
        return status

    def flush(self) -> int:
        """依序重送佇列；遇到第一個送不出去的就停（保持順序）。回傳送出幾筆。"""
        n = 0
        while self.queue:
            status = self._post(self.queue[0])
            if status == "queued":
                break
            self.queue.pop(0)
            n += 1
        return n


# ---------------------------------------------------------------- 管線


@dataclass
class PipeConfig:
    source: str                     # "pose" | "defect"
    site: str
    camera_id: str
    prefix: str                     # event_id 前綴，例如 s1cam2
    threshold: float
    model: dict
    cooldown_s: float = 60.0        # V22：同一支攝影機告警後 60 秒內不再送


@dataclass
class Pipeline:
    source: Source
    detector: Detector
    sender: ApiSender
    cfg: PipeConfig
    last_alert: datetime | None = None
    log: list[dict] = field(default_factory=list)
    _same_sec: dict = field(default_factory=dict)

    def step(self) -> str:
        """跑一張：回傳 below / cooldown / sent / duplicate / rejected / queued。"""
        fr = self.source.grab()
        out = self.detector.predict(fr.image)
        score = float(out["score"])
        if score < self.cfg.threshold:
            status = "below"
        elif self.last_alert is not None and (fr.t - self.last_alert).total_seconds() < self.cfg.cooldown_s:
            status = "cooldown"
        else:
            key = fr.t.replace(microsecond=0)
            n = self._same_sec.get(key, 0)
            self._same_sec[key] = n + 1
            ev = {"event_id": f"{self.cfg.prefix}-{fr.t:%Y%m%d-%H%M%S}-{n}", "source": self.cfg.source,
                  "site": self.cfg.site, "camera_id": self.cfg.camera_id,
                  "event_time": fr.t.isoformat(timespec="seconds"),
                  "kind": {"pose": "fall", "defect": "ng"}[self.cfg.source],
                  "score": round(score, 3), "threshold": self.cfg.threshold,
                  "evidence": out.get("evidence", {}), "model": self.cfg.model}
            status = self.sender.send(ev)
            if status != "rejected":
                self.last_alert = fr.t
        self.log.append({"t": fr.t.isoformat(timespec="seconds"), "score": round(score, 3), "status": status})
        return status


class StepClock:
    """測試用時鐘：每呼叫一次往前走 step 秒。"""

    def __init__(self, start: datetime, step_s: float = 3.0):
        self.t, self.step = start - timedelta(seconds=step_s), timedelta(seconds=step_s)

    def __call__(self) -> datetime:
        self.t += self.step
        return self.t
