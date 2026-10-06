# P16｜跌倒偵測的門檻、錯誤與測試集

以成本 10×漏報＋1×誤報 在 train（S01–S18）依受試者 5 折的 out-of-fold 序列分數上選門檻；錯誤分析；測試集 S19–S24 打開一次；產生全課定版的跌倒模型與設定檔。

    python P16-pose-eval/build_fall_model.py   # 產生定版檔（約 10 秒）
    python P16-pose-eval/evaluate.py           # 讀定版檔：選門檻、錯誤分析、測試集
    python P16-pose-eval/pitfalls.py           # AI 常見錯：在測試集上挑門檻、挑特徵

## 定版檔（V29 讀這兩個檔）

| 檔案 | 內容 |
| --- | --- |
| `demo/models/fall-hgb.joblib` | `HistGradientBoostingClassifier(random_state=0)`，10 個 `common.WINDOW_FEATURES`，S01–S18 全部重新訓練；逐格輸出「正在跌」機率，序列分數取最大值 |
| `demo/config/fall.yaml` | 模型 sha256_12、sklearn 版本、特徵、標籤定義、訓練資料 hash、門檻（值、依據、核准狀態）、out-of-fold 與測試集結果 |

`build_fall_model.py` 把 `OMP_NUM_THREADS` 設成 1：HGB 用不同執行緒數訓練，模型檔的 hash 會不同；固定 1 執行緒，重跑會得到逐位元組相同的檔案。

使用方式（V29）：

```python
import joblib, yaml
from common import window_features
cfg = yaml.safe_load(open("config/fall.yaml", encoding="utf-8"))
model = joblib.load(cfg["model"]["path"])
p = model.predict_proba(window_features(df)[cfg["model"]["features"]])[:, 1]
alarm = p.max() >= cfg["threshold"]["value"]    # 一段序列
```
