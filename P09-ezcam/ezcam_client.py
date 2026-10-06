"""P09｜EZCam client：要求拍照 → 輪詢最新影像 → 下載 JPEG → numpy 陣列。

    from ezcam_client import capture
    shot = capture()          # NTUNHS_MOCK=1 時讀 data/photos/ezcam-sample.jpg，不連網路
    shot.image                # (240, 320, 3) uint8，BGR

真實模式需要兩個環境變數（金鑰不可寫進任何檔案）：
    EZCAM_API_KEY   必填，放在 X-API-Key 標頭
    EZCAM_BASE_URL  選填，預設 https://pi5-cam.4impact.cc
API 規格見 hives/4-learn/MCP_SAMPLE.md。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from common import PHOTOS

DEFAULT_BASE = "https://pi5-cam.4impact.cc"
MOCK_PATH = PHOTOS / "ezcam-sample.jpg"


@dataclass
class Shot:
    image: np.ndarray          # HxWx3 uint8，BGR
    ts: str                    # 相機端的上傳時間（字串，到秒）
    jpeg_bytes: int            # 下載的 JPEG 大小
    waited_s: float            # 從要求拍照到拿到新照片花了幾秒
    polls: int                 # 輪詢了幾次 /api/latest
    source: str                # "mock" 或 "ezcam"


def is_mock() -> bool:
    return os.environ.get("NTUNHS_MOCK") == "1"


def decode_jpeg(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"JPEG 解碼失敗（{len(data)} bytes）：可能下載不完整")
    return img


def _mock() -> Shot:
    data = MOCK_PATH.read_bytes()
    return Shot(decode_jpeg(data), "mock", len(data), 0.0, 0, "mock")


def capture(wait_s: float = 15.0, poll_s: float = 1.0) -> Shot:
    """要求拍一張新照片並等它上傳。mock 模式直接讀範例照片。"""
    if is_mock():
        return _mock()
    import httpx

    key = os.environ.get("EZCAM_API_KEY")
    if not key:
        raise RuntimeError("沒有設定環境變數 EZCAM_API_KEY；課堂練習請設 NTUNHS_MOCK=1")
    base = os.environ.get("EZCAM_BASE_URL", DEFAULT_BASE).rstrip("/")
    with httpx.Client(timeout=10.0, headers={"X-API-Key": key}) as c:
        def latest() -> dict:
            r = c.get(f"{base}/api/latest")
            if r.status_code == 401:
                raise PermissionError("EZCam 回 401：EZCAM_API_KEY 錯誤或過期")
            if r.status_code in (502, 503, 504, 530):
                raise ConnectionError(f"EZCam 回 {r.status_code}：Pi5 或 Cloudflare tunnel 沒在線上")
            r.raise_for_status()
            return r.json()

        before = latest().get("ts")                    # 先記下「舊照片」的時間
        c.post(f"{base}/api/capture").raise_for_status()
        t0, polls = time.monotonic(), 0
        while time.monotonic() - t0 < wait_s:
            time.sleep(poll_s)
            info, polls = latest(), polls + 1
            if info.get("has") and info.get("ts") != before:   # ts 變了才是新照片
                # current.jpg 每次被覆蓋，加 query string 避免拿到快取的舊圖
                r = c.get(info["url"], params={"t": f"{time.time():.0f}"})
                r.raise_for_status()
                return Shot(decode_jpeg(r.content), info["ts"], len(r.content),
                            round(time.monotonic() - t0, 1), polls, "ezcam")
    raise TimeoutError(f"{wait_s:.0f} 秒內沒有新照片：相機可能沒開、沒連上網路，或沒在輪詢")


if __name__ == "__main__":
    s = capture()
    print(f"來源 {s.source}  ts {s.ts}  shape {s.image.shape}  JPEG {s.jpeg_bytes} bytes  "
          f"等待 {s.waited_s} 秒  輪詢 {s.polls} 次")
