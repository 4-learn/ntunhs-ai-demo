"""P09｜EZCam：從攝影機到陣列。mock 取像、320×240 的像素預算、3 秒輪詢的時間軸、對跌倒偵測的影響。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import FPS, PHOTOS, load_poses       # noqa: E402
from ezcam_client import capture, is_mock         # noqa: E402

OUT = Path("/tmp/ntunhs/P09")
OUT.mkdir(parents=True, exist_ok=True)

print("== 1. 取一張：capture() 回傳陣列 ==")
shot = capture()
img = shot.image
print(f"mock 模式 {is_mock()}；來源 {shot.source}")
print(f"shape {img.shape}  dtype {img.dtype}  JPEG {shot.jpeg_bytes} bytes")
print(f"每像素 {shot.jpeg_bytes * 8 / (img.shape[0] * img.shape[1]):.2f} bit（未壓縮是 24 bit）")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"灰階平均 {gray.mean():.1f}；1% / 99% 分位 {np.percentile(gray, 1):.0f} / {np.percentile(gray, 99):.0f}")
print(f"清晰度（拉普拉斯變異數，P06）{cv2.Laplacian(gray, cv2.CV_32F).var():.1f}")

print("\n== 2. 320×240 的像素預算 ==")
room = cv2.imread(str(PHOTOS / "care-room.png"))
s = 320 / room.shape[1]
person_h = 665 - 270                               # P05：扶助行器的長者在 1448 寬原圖約 395 像素高
print(f"寢室照片縮成 320×240：長者高 {person_h * s:.0f} 像素")
for part, frac in [("頭（約身高 1/8）", 1 / 8), ("前臂（約 1/6）", 1 / 6), ("手掌寬（約 1/20）", 1 / 20)]:
    print(f"  {part:<12} 約 {person_h * s * frac:4.1f} 像素")
small = cv2.resize(room, (320, 240), interpolation=cv2.INTER_AREA)
ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
print(f"寢室照片 320×240、JPEG 品質 60：{len(enc)} bytes")
up = cv2.resize(small, (1448, 1086), interpolation=cv2.INTER_CUBIC)
print(f"再放大回 1448×1086，和原圖平均差 {cv2.absdiff(up, room).mean():.2f}：放大補不回細節")
cv2.imwrite(str(OUT / "room-320.jpg"), small)

print("\n== 3. 一次 capture() 要等多久（模擬，相機每 3 秒輪詢）==")
rng = np.random.default_rng(0)
n = 1000
check_phase = rng.uniform(0, 3, n)                 # 要求拍照的瞬間，離相機下一次輪詢還有幾秒
upload = rng.uniform(0.3, 1.0, n)                  # 拍照＋上傳（假設值，未實測）
poll = np.ceil(check_phase + upload)               # client 每 1 秒問一次 /api/latest
print(f"相機看到要求：平均 {check_phase.mean():.2f} 秒，最慢 {check_phase.max():.2f} 秒")
v, c = np.unique(poll, return_counts=True)
print(f"client 拿到新照片：平均 {poll.mean():.2f} 秒")
print("  分布：", "  ".join(f"{int(a)} 秒 {b / n:.3f}" for a, b in zip(v, c)))
print("連續拍：每張至少隔一次輪詢 → 最快約每 3 秒 1 張，不是每秒 10 張的串流")

print("\n== 4. 跌倒在 10 fps 與 3 秒一張下長什麼樣 ==")
df = load_poses()
falls = df[df["activity"] == "fall"]
seq = falls[falls["seq_id"] == "SQ0012"]
print(f"資料是 {FPS} fps（P13 起用的骨架資料，合成）；一段 {len(seq)} 格 = {len(seq) / FPS:.0f} 秒")
code = {"standing": "站", "sitting": "坐", "bending": "彎", "lying": "躺"}
print("SQ0012 每 0.1 秒：", "".join(code[p] for p in seq["posture"]))
print("SQ0012 每 3 秒  ：", "".join(code[p] for p in seq["posture"].iloc[::3 * FPS]), "（第 0、3 秒）")

print("\n== 5. 3 秒一張，抓得到「正在倒」的那一刻嗎？ ==")
rows = []
for sid, g in falls.groupby("seq_id"):
    post = g["posture"].to_numpy()
    start = int(g["fall_frame"].iloc[0])
    change = int(np.argmax(post != post[0])) if (post != post[0]).any() else len(post)
    for phase in range(3 * FPS):                   # 第一張落在第 phase 格，之後每 30 格一張
        idx = np.arange(phase, len(post), 3 * FPS)
        rows.append({"seq": sid, "phase": phase, "張數": len(idx),
                     "拍到倒下過程": bool(((idx >= start) & (idx < change)).any()),
                     "看得到姿勢改變": len(set(post[idx])) > 1, "結尾": post[-1]})
r = pd.DataFrame(rows)
print(f"跌倒片段 {r['seq'].nunique()} 段 × 起拍時間 {3 * FPS} 種 = {len(r)} 種情形")
print("每段拍到幾張：", "  ".join(f"{k} 張 {v} 種" for k, v in r["張數"].value_counts().sort_index().items()))
print(f"拍到「倒下過程」（開始倒 → 姿勢改變之間）的比例 {r['拍到倒下過程'].mean():.3f}")
print(f"前後兩張姿勢不同的比例 {r['看得到姿勢改變'].mean():.3f}")
dur = [int(np.argmax(g["posture"].to_numpy() != "standing")) - int(g["fall_frame"].iloc[0])
       for _, g in falls.groupby("seq_id")]
print(f"「開始倒 → 姿勢改變」只有 {min(dur) / FPS:.1f}–{max(dur) / FPS:.1f} 秒")
print("倒完的姿勢：", "  ".join(f"{k} {v} 段" for k, v in falls.groupby("seq_id")["posture"].last().value_counts().items()))
sit = df[df["activity"] == "sit_down"].groupby("seq_id")["posture"].agg(lambda p: p.iloc[0] + "→" + p.iloc[-1])
fall_sit = falls.groupby("seq_id")["posture"].agg(lambda p: p.iloc[0] + "→" + p.iloc[-1])
print(f"正常坐下 {len(sit)} 段，頭尾姿勢：", "  ".join(f"{k} {v} 段" for k, v in sit.value_counts().items()))
print(f"跌倒後坐在地上 {int((fall_sit == 'standing→sitting').sum())} 段：頭尾姿勢也是 standing→sitting")
