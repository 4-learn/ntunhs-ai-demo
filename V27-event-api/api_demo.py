"""V27 示範：Event 模型驗證、事件 API 三個端點（TestClient，不起真的 server）。

    python V27-event-api/api_demo.py
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import ValidationError

from event_api import EXAMPLE_FALL, EXAMPLE_NG, Event, create_app, make_client


def errors(d):
    """驗證一筆 dict，回傳「欄位: 訊息」清單。"""
    try:
        Event.model_validate(d)
        return ["通過"]
    except ValidationError as e:
        return [f"{'.'.join(map(str, x['loc'])) or '(整筆)'}: {x['msg']}" for x in e.errors()]


print("== 1. 兩組的範例事件都通過驗證 ==")
for name, d in [("健康組 fall", EXAMPLE_FALL), ("製造組 ng", EXAMPLE_NG)]:
    ev = Event.model_validate(d)
    print(f"{name}: {ev.event_id}  score {ev.score:.3f} ≥ 門檻 {ev.threshold:.3f}  模型 {ev.model.name}@{ev.model.sha256_12}")

print("\n== 2. 錯誤欄位：pydantic 告訴你錯在哪 ==")
bad = {
    "少了 model": {k: v for k, v in EXAMPLE_FALL.items() if k != "model"},
    "score 寫成百分比": {**EXAMPLE_FALL, "score": 91},
    "時間沒有時區": {**EXAMPLE_FALL, "event_time": "2026-01-12 02:15:03"},
    "kind 和 source 不合": {**EXAMPLE_FALL, "kind": "ng"},
    "evidence 塞了影像": {**EXAMPLE_NG, "evidence": {"part_id": "P0207", "image_b64": "iVBORw0KGgo" * 40}},
    "多一個拼錯的欄位": {**EXAMPLE_FALL, "camara_id": "cam3"},
    "低於門檻還送出": {**EXAMPLE_FALL, "score": 0.90},
}
for name, d in bad.items():
    print(f"[{name}]")
    for line in errors(d):
        print("  ", line)

print("\n== 3. 事件 API：POST /events ==")
client = make_client(create_app())          # SQLite :memory:，每次都是乾淨的
for d in (EXAMPLE_FALL, EXAMPLE_NG):
    r = client.post("/events", json=d)
    print(r.status_code, r.json())
r = client.post("/events", json=EXAMPLE_FALL)
print("同一筆再送一次：", r.status_code, r.json()["detail"])
r = client.post("/events", json={**EXAMPLE_FALL, "event_id": "s1cam2-20260112-021503-1", "score": 91})
print("score=91：", r.status_code, r.json()["detail"][0]["loc"], r.json()["detail"][0]["type"])

# 再補幾筆，讓查詢有東西可篩
more = []
for i, (t, s) in enumerate([("2026-01-12T03:40:10+08:00", 0.96), ("2026-01-13T01:05:44+08:00", 0.99)]):
    d = copy.deepcopy(EXAMPLE_FALL)
    d.update(event_id=f"s1cam1-{t[:10].replace('-', '')}-{t[11:19].replace(':', '')}-0", camera_id="cam1",
             event_time=t, score=s)
    more.append(client.post("/events", json=d).status_code)
print("再補 2 筆跌倒事件：", more)

print("\n== 4. GET /events?kind=&since= ==")
print("全部：", len(client.get("/events").json()), "筆")
fall = client.get("/events", params={"kind": "fall"}).json()
print("kind=fall：", [e["event_id"] for e in fall])
since = client.get("/events", params={"kind": "fall", "since": "2026-01-12T03:00:00+08:00"}).json()
print("kind=fall, since 01-12 03:00：", [e["event_id"] for e in since])
utc = client.get("/events", params={"since": "2026-01-11T19:00:00Z"}).json()
print("since 2026-01-11T19:00:00Z（＝台灣 01-12 03:00）：", len(utc), "筆")
r = client.get("/events", params={"since": "2026-01-12T03:00:00"})
print("since 沒有時區：", r.status_code)

print("\n== 5. POST /events/{id}/ack：人確認 ==")
eid = EXAMPLE_FALL["event_id"]
r = client.post(f"/events/{eid}/ack", json={"by": "N02", "verdict": "true_alarm"})
print(r.status_code, r.json())
r = client.post(f"/events/{eid}/ack", json={"by": "N05", "verdict": "false_alarm"})
print("第二個人再按：", r.status_code, r.json()["detail"])
r = client.post("/events/no-such-event/ack", json={"by": "N02", "verdict": "true_alarm"})
print("不存在的事件：", r.status_code)
r = client.post(f"/events/{EXAMPLE_NG['event_id']}/ack", json={"by": "Q01", "verdict": "maybe"})
print("verdict=maybe：", r.status_code, r.json()["detail"][0]["loc"])
acked = [e for e in client.get("/events").json() if e["ack"]]
print("已確認：", [(e["event_id"], e["ack"]["by"], e["ack"]["verdict"]) for e in acked])
