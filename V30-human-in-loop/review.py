"""V30 示範：把模擬一週的事件送進事件 API、重播值班人員的確認，再從 API 讀回來統計。

    python V30-human-in-loop/review.py
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT

sys.path.insert(0, str(ROOT / "V27-event-api"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_api import create_app, make_client  # noqa: E402
from sim_week import simulate_week  # noqa: E402

pd.set_option("display.unicode.east_asian_width", True)


class Clock:
    """可以撥的時鐘：重播 ack 時，讓 API 記下「當時」的時間，而不是現在。"""
    t = datetime(2026, 1, 12)

    def __call__(self):
        return self.t


clock = Clock()
client = make_client(create_app(now=clock))
events, acks = simulate_week(seed=0)
codes = [client.post("/events", json=e).status_code for e in events]
for eid, (by, verdict, t) in sorted(acks.items(), key=lambda kv: kv[1][2]):
    clock.t = t
    client.post(f"/events/{eid}/ack", json={"by": by, "verdict": verdict})

rows = client.get("/events").json()
df = pd.DataFrame([{"event_id": e["event_id"], "kind": e["kind"], "camera": e["camera_id"],
                    "time": pd.Timestamp(e["event_time"]), "score": e["score"], "threshold": e["threshold"],
                    "by": e["ack"]["by"] if e["ack"] else None,
                    "verdict": e["ack"]["verdict"] if e["ack"] else None,
                    "delay_min": ((pd.Timestamp(e["ack"]["at"]) - pd.Timestamp(e["event_time"])).total_seconds() / 60)
                    if e["ack"] else np.nan}
                   for e in rows])
df["night"] = (df["time"].dt.hour < 7) | (df["time"].dt.hour >= 22)

print("== 1. 一週的事件與確認（模擬資料）==")
print(f"送出 {len(codes)} 筆，201 共 {codes.count(201)} 筆；有人確認 {df['verdict'].notna().sum()} 筆")


def rate_table(g):
    return pd.DataFrame({"事件": g.size(),
                         "已確認": g["verdict"].count(),
                         "真告警": g["verdict"].apply(lambda s: (s == "true_alarm").sum()),
                         "誤報": g["verdict"].apply(lambda s: (s == "false_alarm").sum())}).assign(
        真告警率=lambda t: (t["真告警"] / t["已確認"]).round(3),
        未確認率=lambda t: (1 - t["已確認"] / t["事件"]).round(3))


print("\n== 2. 真告警率 = 真告警 ÷ 已確認（未確認的不算進分母）==")
print(rate_table(df.groupby("kind")).to_string())
print("\n健康組，依攝影機：")
print(rate_table(df[df["kind"] == "fall"].groupby("camera")).to_string())
print("\n健康組，夜班（22–07 點）vs 白天：")
fall = df[df["kind"] == "fall"]
print(rate_table(fall.groupby(fall["night"].map({True: "夜班", False: "白天"}))).to_string())

print("\n== 3. 如果把門檻調高：已確認事件上的「what-if」==")
acked = df[df["verdict"].notna()]


def what_if(sub, grid, miss_cost):
    """只能往「更高」的門檻推算：低於現行門檻的分數根本沒變成事件，看不到。"""
    out = []
    for t in grid:
        keep = sub["score"] >= t
        lost_true = int(((~keep) & (sub["verdict"] == "true_alarm")).sum())
        fa = int((keep & (sub["verdict"] == "false_alarm")).sum())
        out.append({"門檻": f"{t:.2f}", "仍會通知": int(keep.sum()),
                    "少掉的誤報": int(((~keep) & (sub["verdict"] == "false_alarm")).sum()),
                    "變成漏報": lost_true, "成本": miss_cost * lost_true + fa})
    return pd.DataFrame(out)


cam1 = acked[(acked["kind"] == "fall") & (acked["camera"] == "cam1")]
print("健康組 cam1（寢室），成本 = 10×漏報 + 1×誤報：")
print(what_if(cam1, [0.95, 0.96, 0.97, 0.98, 0.99], 10).to_string(index=False))
print("真告警的最低分數：", f"{cam1.loc[cam1['verdict'] == 'true_alarm', 'score'].min():.3f}")
ng = acked[acked["kind"] == "ng"]
print("\n製造組 line1cam1，成本 = 20×漏報 + 1×誤報：")
print(what_if(ng, [0.62, 0.66, 0.70, 0.74], 20).to_string(index=False))
print("真瑕疵的最低分數：", f"{ng.loc[ng['verdict'] == 'true_alarm', 'score'].min():.3f}")

print("\n== 4. 通知分級：護理站螢幕 vs 手機推播 ==")
HIGH = 0.98      # 分數高 → 立刻推播
ESCALATE = 5     # 分數低、但 5 分鐘內沒人確認 → 升級推播
fall = df[df["kind"] == "fall"].copy()
fall["立刻推播"] = fall["score"] >= HIGH
fall["升級推播"] = ~fall["立刻推播"] & ~(fall["delay_min"] <= ESCALATE)
tab = pd.DataFrame({
    "全部推播": [len(fall), int((fall["verdict"] == "true_alarm").sum())],
    "只推高分": [int(fall["立刻推播"].sum()), int((fall["立刻推播"] & (fall["verdict"] == "true_alarm")).sum())],
    "高分＋升級": [int((fall["立刻推播"] | fall["升級推播"]).sum()),
               int(((fall["立刻推播"] | fall["升級推播"]) & (fall["verdict"] == "true_alarm")).sum())],
}, index=["手機響幾次", "其中真跌倒"])
print(tab.to_string())
lost = fall[(fall["verdict"] == "true_alarm") & ~fall["立刻推播"]]
print("只推高分時，只上護理站螢幕的真跌倒：", lost["event_id"].tolist())
nights = fall[fall["night"]]
print(f"夜班手機一週響：全部推播 {len(nights)} 次；高分＋升級 {int((nights['立刻推播'] | nights['升級推播']).sum())} 次")

print("\n== 5. 門檻建議報告（只產出建議，不改設定檔）==")
MIN_TRUE = 10      # 已確認真告警少於這個數，不給調整建議
MARGIN = 0.05      # 新門檻至少比「最低分的真告警」低這麼多


def suggest(name, sub, current, grid):
    tr = sub.loc[sub["verdict"] == "true_alarm", "score"]
    fa = sub["verdict"] == "false_alarm"
    lines = [f"[{name}] 現行門檻 {current:.2f}；本週已確認 {len(sub)} 筆（真告警 {len(tr)}、誤報 {int(fa.sum())}）"]
    if len(tr) < MIN_TRUE:
        lines.append(f"  建議：維持 {current:.2f}。真告警只有 {len(tr)} 筆（< {MIN_TRUE}），樣本不足以判斷調高會漏掉什麼。")
        lines.append("  下一步：繼續收集確認紀錄；cam1 誤報集中，先查現場（床的位置、鏡頭角度），不是先動門檻。")
        return lines
    ok = [t for t in grid if t <= tr.min() - MARGIN]
    new = max(ok) if ok else current
    keep = sub["score"] >= new
    lines.append(f"  真告警最低分 {tr.min():.3f}；保留 {MARGIN:.2f} 餘裕 → 候選門檻 {new:.2f}")
    lines.append(f"  本週若用 {new:.2f}：誤報 {int(fa.sum())} → {int((keep & fa).sum())}，已確認的真告警不會少")
    lines.append(f"  風險：低於 {current:.2f} 的分數從來沒有變成事件，這份紀錄看不到漏報；要再用 V12 的驗證集重算一次")
    lines.append("  狀態：待核准（核准前程式繼續使用現行門檻）")
    return lines


for line in suggest("健康組 cam1", cam1, 0.95, [0.96, 0.97, 0.98, 0.99]):
    print(line)
for line in suggest("製造組 line1cam1", ng, 0.62, [0.66, 0.70, 0.74]):
    print(line)
