# P14｜依人分組切割與姿勢分類

隨機切格子 vs 依受試者分組（StratifiedGroupKFold）的分數差異；HGB 姿勢四類（fit＝S01–S14、valid＝S15–S18）、混淆矩陣、錯誤集中在姿勢轉換、bending 最難。測試集 S19–S24 不碰。

    python P14-pose-classify/split_compare.py
    python P14-pose-classify/classify.py

腳本把 `OMP_NUM_THREADS` 設成 1：HGB 結果可重現，在多人共用的機器上也比較快。
