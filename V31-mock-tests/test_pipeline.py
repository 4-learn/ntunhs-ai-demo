"""V31｜沒有攝影機也能驗收：管線三段各自換成假的，用 TestClient 打 V27 的事件 API。

    python V31-mock-tests/run_tests.py          # 講義用：只印每個測試的結果與通過數
    python -m pytest V31-mock-tests -q          # 平常開發用
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "V27-event-api"))
sys.path.insert(0, str(HERE.parent))          # demo/（common.py）

from event_api import create_app, make_client  # noqa: E402
from pipeline import (TW, ApiSender, FakeDetector, MockCamera, PipeConfig, Pipeline,  # noqa: E402
                      StepClock, make_source)

MODEL = {"name": "fall-hgb", "version": "2026.01", "sha256_12": "05ba1c2d31cf"}
CFG = PipeConfig(source="pose", site="松年本館", camera_id="cam2", prefix="s1cam2",
                 threshold=0.95, model=MODEL, cooldown_s=60)
START = datetime(2026, 1, 12, 2, 15, 0, tzinfo=TW)


@pytest.fixture
def client():
    return make_client(create_app())          # 每個測試一個全新的 :memory: 資料庫


def build(client, scores, step_s=3.0, sender=None):
    clock = StepClock(START, step_s)
    return Pipeline(MockCamera(clock), FakeDetector(scores, {"seq_id": "SQ0123", "keypoints_only": True}),
                    sender or ApiSender(client), CFG)


def test_mock_mode_never_touches_network(monkeypatch):
    monkeypatch.setenv("NTUNHS_MOCK", "1")
    monkeypatch.delenv("EZCAM_API_KEY", raising=False)
    import httpx
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: pytest.fail("mock 模式不該連網路"))
    fr = make_source().grab()
    assert isinstance(make_source(), MockCamera)
    assert fr.image.shape == (240, 320, 3) and fr.t.tzinfo is not None


def test_real_camera_needs_key(monkeypatch):
    monkeypatch.setenv("NTUNHS_MOCK", "0")
    monkeypatch.delenv("EZCAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EZCAM_API_KEY"):
        make_source()


def test_below_threshold_sends_nothing(client):
    pipe = build(client, [0.10, 0.949, 0.50])
    assert [pipe.step() for _ in range(3)] == ["below"] * 3
    assert client.get("/events").json() == []


def test_alarm_becomes_valid_event(client):
    pipe = build(client, [0.10, 0.98])
    assert [pipe.step(), pipe.step()] == ["below", "sent"]
    (ev,) = client.get("/events", params={"kind": "fall"}).json()
    assert ev["event_id"] == "s1cam2-20260112-021503-0"
    assert ev["event_time"] == "2026-01-12T02:15:03+08:00"
    assert ev["score"] == 0.98 and ev["model"]["sha256_12"] == MODEL["sha256_12"]
    assert "image" not in ev["evidence"] and ev["ack"] is None


def test_cooldown_60s(client):
    pipe = build(client, [0.98], step_s=10)          # 每 10 秒一張，分數一直很高
    got = [pipe.step() for _ in range(8)]           # 0, 10, ..., 70 秒
    assert got == ["sent"] + ["cooldown"] * 5 + ["sent", "cooldown"]
    assert len(client.get("/events").json()) == 2


def test_resend_is_idempotent(client):
    sender = ApiSender(client)
    pipe = build(client, [0.98], sender=sender)
    pipe.step()
    ev = client.get("/events").json()[0]
    ev.pop("ack")
    assert sender.send(ev) == "duplicate"           # 網路逾時後重送同一筆
    assert len(client.get("/events").json()) == 1


class FlakyClient:
    """前 n 次 POST 直接丟連線錯誤，之後才轉給真的 TestClient。"""

    def __init__(self, inner, fail_times):
        self.inner, self.fail = inner, fail_times

    def post(self, *a, **k):
        if self.fail > 0:
            self.fail -= 1
            raise ConnectionError("API 暫時連不上")
        return self.inner.post(*a, **k)


def test_api_down_then_flush_in_order(client):
    sender = ApiSender(FlakyClient(client, fail_times=2))
    pipe = build(client, [0.98], step_s=61, sender=sender)
    assert [pipe.step() for _ in range(3)] == ["queued", "queued", "sent"]
    ids = [e["event_id"] for e in client.get("/events").json()]
    assert ids == ["s1cam2-20260112-021500-0", "s1cam2-20260112-021601-0", "s1cam2-20260112-021702-0"]
    assert sender.queue == []


def test_bad_score_is_rejected_not_retried(client):
    sender = ApiSender(client)
    pipe = build(client, [91.0], sender=sender)     # 推論器回傳百分比
    assert pipe.step() == "rejected"
    assert sender.queue == [] and sender.rejected == [("s1cam2-20260112-021500-0", "score")]
    assert pipe.last_alert is None                  # 被拒的不算告警，不進冷卻
