"""V35 示範：產生兩組的交接包，再用「接手方」的角度把每一個 hash 重算一次。

    python V35-handoff/handoff.py        # 寫出 V35-handoff/handoff/{defect,fall}/

需要先有：V28 的記憶庫（cache/models/，沒有會自動建）、P16 的 fall-hgb、P18 的 fall-handoff.json。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_THREADS", "2")
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_lib import OUT, P18_BLOCKS, build_defect, fall_pack, verify, write_pack  # noqa: E402

shutil.rmtree(OUT, ignore_errors=True)

print("== 1. 產生交接包 ==")
packs = {"defect": build_defect("metal_nut"), "fall": fall_pack()}
for g, (pack, eval_list) in packs.items():
    d = write_pack(g, pack, eval_list)
    extra = [k for k in pack if k not in P18_BLOCKS]
    print(f"{g}：{sorted(p.name for p in d.iterdir())}；P18 區塊 {sum(k in pack for k in P18_BLOCKS)}/{len(P18_BLOCKS)}，"
          f"另加 {extra}")

print("\n== 2. SHA256SUMS（接手方可以直接 sha256sum -c）==")
for g in packs:
    for line in (OUT / g / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        h, n = line.split("  ")
        print(f"{g}/{n}：{h[:12]}")

print("\n== 3. 接手方驗證：每一個 hash 都重算 ==")
for g in packs:
    print(f"[{g}]")
    for item, ok, msg in verify(OUT / g):
        print(f"  {'一致' if ok else '不一致'}  {item}" + (f"（{msg}）" if msg and not ok else ""))

print("\n== 4. 兩份交接包的重點 ==")
for g, (pack, _) in packs.items():
    e, s = pack["eval_set"], pack["scores"]["test"]
    n = e.get("items", e.get("sequences"))
    print(f"{g}：評估集 {n} 件、清單 sha256_12 {e['list_sha256'][:12]}；門檻 {pack['threshold']['value']}"
          f"（{pack['threshold']['approved_by']}）；測試 tp {s['tp']} fn {s['fn']} fp {s['fp']}")
    print(f"   模型 {', '.join(m['name'] + ' ' + m['sha256_12'] for m in pack['models'])}")
    print(f"   已知錯誤 {['/'.join(x['item'].split('/')[-2:]) + ' ' + x['kind'] for x in pack['known_model_errors']]}"
          f"；待辦 {len(pack['open_issues'])} 項")

print("\n== 5. 改了門檻（0.95 → 0.9），會被哪一項抓到 ==")
tmp = Path("/tmp/ntunhs/V35/tampered")
shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(OUT / "fall", tmp)
p = json.loads((tmp / "handoff.json").read_text(encoding="utf-8"))
p["threshold"]["value"] = 0.9
(tmp / "handoff.json").write_text(json.dumps(p, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("只改 handoff.json：", [item for item, ok, _ in verify(tmp) if not ok])
write_pack_dir = tmp.parent / "fall"
shutil.rmtree(write_pack_dir, ignore_errors=True)
import handoff_lib  # noqa: E402

handoff_lib.OUT = tmp.parent
write_pack("fall", p, (tmp / "eval-list.csv").read_text(encoding="utf-8"))      # 連 SHA256SUMS 一起重算
print("連 SHA256SUMS 一起重算：", [(item, msg) for item, ok, msg in verify(write_pack_dir) if not ok])
