"""把 P12 的三張照片推論成 poses.csv 格式（P13 與後續各節共用的小工具）。"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")
import pandas as pd

from common import KPTS, MODELS, PHOTOS

PHOTO_NAMES = ["care-room", "care-corridor-fall", "factory-lifting"]
# 教師目視照片後寫下的「這個人在做什麼」（YOLO 依框信心排序，第 0 人＝信心最高者）
WHAT = {("care-room", 0): "坐在椅子上", ("care-room", 1): "扶助行器站著",
        ("care-corridor-fall", 0): "倒在地上", ("care-corridor-fall", 1): "照服員走來",
        ("factory-lifting", 0): "彎腰搬箱", ("factory-lifting", 1): "站在工作台前"}


def photo_table() -> pd.DataFrame:
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)
    model = YOLO(str(MODELS / "yolo11n-pose.pt"))
    recs = []
    for name in PHOTO_NAMES:
        r = model(str(PHOTOS / f"{name}.png"), verbose=False)[0]
        xy = r.keypoints.xy.numpy().astype(float)
        conf = r.keypoints.conf.numpy().astype(float)
        for i in range(len(xy)):
            rec = {"photo": name, "person": i, "what": WHAT.get((name, i), "")}
            for k, n in enumerate(KPTS):
                rec[f"{n}_x"] = round(xy[i, k, 0], 1)
                rec[f"{n}_y"] = round(xy[i, k, 1], 1)
                rec[f"{n}_c"] = round(conf[i, k], 3)
            recs.append(rec)
    return pd.DataFrame(recs)
