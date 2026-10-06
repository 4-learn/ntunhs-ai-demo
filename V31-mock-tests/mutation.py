"""V31｜測試會不會「抓到錯」：故意把管線改壞（mutant），看兩份測試各有幾個失敗。

    python V31-mock-tests/mutation.py

mutant A：冷卻失效（每次都當作沒告警過）
mutant B：門檻比較寫反（低分送、高分不送）
"""
import contextlib
import io
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["NTUNHS_MOCK"] = "1"

import pipeline  # noqa: E402

ORIGINAL = pipeline.Pipeline.step


def no_cooldown(self):
    self.last_alert = None
    return ORIGINAL(self)


class Flipped:
    """把推論器的分數翻轉（1 - score）：等同於「門檻比較寫反」。"""

    def __init__(self, inner):
        self.inner = inner

    def predict(self, image):
        out = self.inner.predict(image)
        return {**out, "score": round(1 - out["score"], 3)}


def flipped(self):
    if not isinstance(self.detector, Flipped):
        self.detector = Flipped(self.detector)
    return ORIGINAL(self)


class Count:
    def __init__(self):
        self.passed = self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.passed += report.outcome == "passed"
            self.failed += report.outcome == "failed"


def run(test_file):
    c = Count()
    for m in [m for m in sys.modules if m in ("test_pipeline", "ai_written_tests")]:
        del sys.modules[m]
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.main([str(HERE / test_file), "-q", "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning"],
                    plugins=[c])
    return f"{c.passed} passed, {c.failed} failed"


import pandas as pd  # noqa: E402

pd.set_option("display.unicode.east_asian_width", True)
rows = []
for name, step in [("原版", ORIGINAL), ("A 冷卻失效", no_cooldown), ("B 門檻反了", flipped)]:
    pipeline.Pipeline.step = step
    rows.append({"管線版本": name, "本節的測試": run("test_pipeline.py"), "AI 寫的測試": run("ai_written_tests.py")})
pipeline.Pipeline.step = ORIGINAL
print("== 同一份管線，兩份測試：改壞之後，誰會失敗？==")
print(pd.DataFrame(rows).to_string(index=False))
