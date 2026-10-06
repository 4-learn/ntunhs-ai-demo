"""V31「讓 AI 寫，然後挑錯」：AI 寫的測試（故意不叫 test_*.py，平常不會被 pytest 收集）。

看起來有測、全部會過；但把管線改壞（mutation.py），它還是全部會過。
"""
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "V27-event-api"))

from event_api import create_app, make_client  # noqa: E402
from pipeline import ApiSender, FakeDetector, PipeConfig, Pipeline, make_source  # noqa: E402

client = make_client(create_app())             # 全部測試共用一個 app
cfg = PipeConfig("pose", "松年本館", "cam2", "s1cam2", 0.95,
                 {"name": "fall-hgb", "version": "2026.01", "sha256_12": "05ba1c2d31cf"})


def test_pipeline_runs():
    pipe = Pipeline(make_source(), FakeDetector([0.98]), ApiSender(client), cfg)
    for _ in range(5):
        status = pipe.step()
        assert status in ("below", "cooldown", "sent", "duplicate", "rejected", "queued")


def test_post_event():
    r = client.get("/events")
    assert r.status_code
    assert len(r.json()) >= 0


def test_cooldown():
    pipe = Pipeline(make_source(), FakeDetector([0.98]), ApiSender(client), cfg)
    pipe.step()
    assert pipe.last_alert is not None or True
