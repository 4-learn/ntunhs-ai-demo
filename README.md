# 國北護 AI 健康科技與智慧製造應用班 — 範例程式

「Python 機器學習（國北護補充 P01–P18）」與「影像 AI 淬鍊實戰專案（V01–V35）」的範例程式與合成資料。
講義在 HackMD：[總課綱](https://hackmd.io/@yillkid/S1YjAnAdzl)｜[機器學習國北護版](https://hackmd.io/c/HyJJtVGoMx)｜[影像 AI](https://hackmd.io/c/H1fytEGozx)

情境：AI 新創「看見科技」接兩個客戶——松年長照中心（健康組，骨架跌倒監測）、精工五金（製造組，瑕疵檢測）。

> 講義與程式為**初稿（2026-10-06）**，尚未實班試教。

## 環境（Python 3.11 以上，CPU 即可）

```bash
python3.11 -m venv ~/.venvs/ml-course
source ~/.venvs/ml-course/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn==1.9.1 pandas==3.0.6 opencv-python matplotlib pytest pillow joblib pyyaml \
            ultralytics onnx onnxruntime fastapi httpx pydantic threadpoolctl
```

**一律從 repo 根目錄、帶 `PYTHONPATH=.` 執行**（避免 import 到別門課的 `common.py`）：

```bash
PYTHONPATH=. python P01-numpy/vectorize.py
NTUNHS_MOCK=1 PYTHONPATH=. python P09-ezcam/ezcam_client.py   # 不連 EZCam，讀 data/photos/ezcam-sample.jpg
```

真的連 EZCam 時，API key 只放環境變數 `EZCAM_API_KEY`，**不要寫進任何檔案**。

## 內容

| 路徑 | 說明 |
| --- | --- |
| `common.py` | 全課共用：讀資料、骨架特徵、窗口特徵、MVTec 讀取與切割 |
| `data/` | **合成**資料（`tools/build_data.py` 產生，`MANIFEST.json` 有 SHA-256）；`data/photos/*.png` 為 AI 生成圖片 |
| `models/` | `fall-hgb.joblib`（P16 定版跌倒模型）、`posture-hgb.joblib`、`yolo11n-pose.pt`／`yolo11n.pt`（Ultralytics 預訓練） |
| `config/fall.yaml` | P16 定版門檻與模型 hash |
| `tools/` | 產生資料、快取、YOLO 瑕疵資料集與訓練 |
| `P*/`、`V*/` | 各節範例（每個資料夾有 README） |

## 製造組需要 MVTec AD（不在 repo 裡）

V01–V18、V24–V28、V33–V35 的製造組範例使用 [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)（**CC BY-NC-SA 4.0，僅限非商業的課堂使用**）。資料由教師另行提供，放在 `~/datasets/mvtec-ad`（或設環境變數 `MVTEC_ROOT`）。

以 MVTec 訓練出的模型與交接包屬衍生物，不放進 repo，第一次使用前由教師產生：

```bash
PYTHONPATH=. python tools/warm_cache.py            # ResNet 特徵快取（首次數分鐘）
PYTHONPATH=. python tools/build_yolo_defects.py    # V15 YOLO 資料集
PYTHONPATH=. python tools/train_yolo_defects.py    # V16 → models/defect-yolo11n.pt（CPU 約 30 分鐘）
PYTHONPATH=. python V14-defect-finetune/train.py   # V14 → models/defect-cls-mnv3s.pt
```

## 授權與資料說明

- 照護、產線、骨架資料全部由程式合成，人員、機構、事件都不是真實資料；合成資料上的分數**不代表**真實場域的表現。
- `ultralytics` 與 YOLO11 權重為 **AGPL-3.0**：若把使用它的服務提供給他人，須依 AGPL 公開對應原始碼，或另購商業授權。
- torchvision 預訓練權重依其模型卡授權；MVTec AD 依 CC BY-NC-SA 4.0。
