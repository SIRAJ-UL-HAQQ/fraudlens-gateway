"""Measurement script for full-train XGBoost fit time and peak memory."""

import ctypes
import time
import tracemalloc
from ctypes import wintypes

from xgboost import XGBClassifier

from ml.src.features import PaySimFeaturePreprocessor
from ml.src.load_data import load_raw_paysim, split_chronological


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def get_peak_working_set_mb() -> float:
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    success = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
    )
    if not success:
        return 0.0
    return pmc.PeakWorkingSetSize / (1024 * 1024)


def measure_fit() -> None:
    print("Loading raw PaySim dataset...")
    df = load_raw_paysim()
    train_df, _, _ = split_chronological(df)

    prep = PaySimFeaturePreprocessor(include_balance_errors=False)
    X_train_trans = prep.fit_transform(train_df)
    y_train = train_df["is_fraud"].to_numpy(dtype=int)

    tracemalloc.start()
    start_time = time.time()

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )
    clf.fit(X_train_trans, y_train)

    elapsed = time.time() - start_time
    _current_trace, peak_trace = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_working_set = get_peak_working_set_mb()

    print(f"Full-train row count: {len(y_train)}")
    print(f"Fit wall-clock elapsed: {elapsed:.2f} seconds")
    print(f"Python traced memory peak: {peak_trace / (1024 * 1024):.2f} MB")
    print(f"Process peak working set (RAM): {peak_working_set:.2f} MB")


if __name__ == "__main__":
    measure_fit()
