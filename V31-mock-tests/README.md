# V31｜Mock 與測試：沒有攝影機也能驗收

管線拆成取像／推論／送 API 三段（`pipeline.py`），`NTUNHS_MOCK=1` 不連 EZCam、不連網路；`test_pipeline.py` 用 TestClient＋假推論器寫 8 個測試，`mutation.py` 故意改壞管線檢驗測試有沒有用。

    python V31-mock-tests/run_tests.py
    python V31-mock-tests/mutation.py
    python -m pytest V31-mock-tests -q -p no:cacheprovider
