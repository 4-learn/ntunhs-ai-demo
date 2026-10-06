"""V31｜在腳本裡呼叫 pytest.main，只印每個測試的結果與通過數（不印秒數，講義輸出才穩定）。

    python V31-mock-tests/run_tests.py
"""
import contextlib
import io
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
os.environ["NTUNHS_MOCK"] = "1"          # 測試時一定是 mock 模式


class Collect:
    """小外掛：記下每個測試的結果。"""

    def __init__(self):
        self.results = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            self.results.append((report.nodeid.split("::")[-1], report.outcome))


col = Collect()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):    # pytest 自己的摘要行含秒數，不印
    code = pytest.main([str(HERE / "test_pipeline.py"), "-q", "-p", "no:cacheprovider",
                        "-W", "ignore::DeprecationWarning"], plugins=[col])

print("== pytest 結果（NTUNHS_MOCK=1，沒有攝影機、沒有網路）==")
for name, outcome in col.results:
    print(f"{outcome.upper():6s} {name}")
passed = sum(o == "passed" for _, o in col.results)
print(f"{passed} passed, {len(col.results) - passed} failed")
if code != 0:
    print(buf.getvalue()[-3000:], file=sys.stderr)
sys.exit(int(code))
