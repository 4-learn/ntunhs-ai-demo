"""P09｜讓 AI 寫，然後挑錯：沒比對 ts 拿到舊照片、沒檢查解碼失敗、金鑰寫死、把輪詢當串流。

全部離線模擬，不連網路。
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import PHOTOS                          # noqa: E402

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


class FakeEZCam:
    """模擬 Pi5 API 的時間軸：要求拍照後，相機要等到下一次輪詢（每 3 秒）才拍，再花 0.6 秒上傳。"""

    def __init__(self, now=0.0, phase=1.8):
        self.now, self.phase = now, phase          # phase：下一次輪詢在幾秒後
        self.photo_ts, self.ready_at = "08:59:57", None

    def capture(self):
        self.ready_at = self.now + self.phase + 0.6

    def latest(self):
        if self.ready_at is not None and self.now >= self.ready_at:
            self.photo_ts = "09:00:02"
        return {"has": True, "ts": self.photo_ts}

    def sleep(self, s):
        self.now += s


print("== 錯誤 1：要求拍照後馬上讀 latest，沒比對 ts ==")
cam = FakeEZCam()
cam.capture()
cam.sleep(1)
print(f"AI 版：1 秒後讀到 ts {cam.latest()['ts']} ← 這是要求拍照「之前」的舊照片")
cam = FakeEZCam()
before = cam.latest()["ts"]
cam.capture()
polls = 0
while cam.latest()["ts"] == before:
    cam.sleep(1)
    polls += 1
print(f"正確版：先記 before={before}，輪詢 {polls} 次後拿到 ts {cam.latest()['ts']}")

print("\n== 錯誤 2：下載不完整，沒檢查 imdecode 的結果 ==")
data = (PHOTOS / "ezcam-sample.jpg").read_bytes()
half = data[: len(data) * 6 // 10]                 # 網路中斷：只收到 60%
img = cv2.imdecode(np.frombuffer(half, np.uint8), cv2.IMREAD_COLOR)
print(f"收到 {len(half)} / {len(data)} bytes，imdecode 回傳 {img}")
try:
    img.shape
except AttributeError as e:
    print("下一行才爆：", e)

print("\n== 錯誤 3：金鑰寫在程式裡 ==")
src = (Path(__file__).parent / "ezcam_client.py").read_text(encoding="utf-8")
print("ezcam_client.py 讀金鑰的方式：", "os.environ.get(\"EZCAM_API_KEY\")" in src)
saved = {k: os.environ.pop(k, None) for k in ("NTUNHS_MOCK", "EZCAM_API_KEY")}
from ezcam_client import capture                    # noqa: E402
try:
    capture()
except RuntimeError as e:
    print("沒設環境變數時：", e)
finally:
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v

print("\n== 錯誤 4：以為 while True: capture() 是每秒 10 張的影片 ==")
rng = np.random.default_rng(0)
t, times = 0.0, []
for _ in range(20):
    wait = (3 - t % 3) + rng.uniform(0.3, 1.0)       # 等下一次輪詢 + 上傳
    t += np.ceil(wait)                              # client 每 1 秒問一次
    times.append(t)
gaps = np.diff([0.0] + times)
print(f"連拍 20 張花了 {t:.0f} 秒；平均每張間隔 {gaps.mean():.2f} 秒 = {1 / gaps.mean():.2f} fps")
print(f"10 fps 的串流同樣 {t:.0f} 秒會有 {t * 10:.0f} 格")
