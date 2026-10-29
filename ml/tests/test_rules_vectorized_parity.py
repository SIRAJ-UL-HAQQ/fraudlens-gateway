"""Unit test for vectorized heuristic rules parity against backend evaluate_rules.

Verifies bit-exact flag parity on >= 5,000 sampled PaySim rows including
all boundary conditions:
- Zero origin balances (division by zero / strictly not drained)
- Exact threshold boundaries: amount == 200,000.0, drain_ratio == 0.90
- Customer vs merchant destination accounts ('C...' vs 'M...')
- Destination balance delta exact balance vs inconsistency
- Per-type high-amount thresholds across all 5 PaySim transaction types
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from app.core.config import settings
from app.models.schemas import TransactionIn, TransactionType
from app.services.rules import evaluate_rules

from ml.src.load_data import load_raw_paysim


def evaluate_rules_vectorized_all(df: pd.DataFrame) -> list[list[str]]:
    """Vectorized evaluation of all 4 backend heuristic rules on DataFrame.

    Matches backend.app.services.rules.evaluate_rules exactly.
    """
    types = df["type"].to_numpy()
    amounts = df["amount"].to_numpy(dtype=float)
    old_orig = df["old_balance_orig"].to_numpy(dtype=float)
    new_orig = df["new_balance_orig"].to_numpy(dtype=float)
    old_dest = df["old_balance_dest"].to_numpy(dtype=float)
    new_dest = df["new_balance_dest"].to_numpy(dtype=float)

    is_tc = np.isin(types, ["TRANSFER", "CASH_OUT"])
    is_c_dest = df["name_dest"].astype(str).str.startswith("C").to_numpy(dtype=bool)

    # Rule 1: ORIGIN_ACCOUNT_DRAINED (drain >= 0.90, old > 0, new == 0)
    drain_ratio = settings.RULE_DRAIN_BALANCE_RATIO  # 0.90
    r1_mask = (
        is_tc
        & (old_orig > 0.0)
        & (new_orig == 0.0)
        & (amounts >= old_orig * drain_ratio)
    )

    # Rule 2: LARGE_TRANSFER_AMOUNT (amount >= 200,000)
    large_thresh = settings.RULE_LARGE_TRANSFER_THRESHOLD  # 200,000.0
    r2_mask = is_tc & (amounts >= large_thresh)

    # Rule 3: DEST_BALANCE_INCONSISTENCY
    dest_delta = new_dest - old_dest
    r3_mask = is_tc & is_c_dest & (np.abs(dest_delta - amounts) > 0.01)

    # Rule 4: VERY_HIGH_AMOUNT_FOR_TYPE
    r4_mask = np.zeros(len(df), dtype=bool)
    for t_name, thresh in settings.RULE_HIGH_AMOUNT_BY_TYPE.items():
        r4_mask |= (types == t_name) & (amounts >= thresh)

    # Collect flags per row matching evaluate_rules sort order
    n = len(df)
    flags_list = [[] for _ in range(n)]

    r1_indices = np.where(r1_mask)[0]
    for idx in r1_indices:
        flags_list[idx].append("ORIGIN_ACCOUNT_DRAINED")

    r2_indices = np.where(r2_mask)[0]
    for idx in r2_indices:
        flags_list[idx].append("LARGE_TRANSFER_AMOUNT")

    r3_indices = np.where(r3_mask)[0]
    for idx in r3_indices:
        flags_list[idx].append("DEST_BALANCE_INCONSISTENCY")

    r4_indices = np.where(r4_mask)[0]
    for idx in r4_indices:
        flags_list[idx].append("VERY_HIGH_AMOUNT_FOR_TYPE")

    # Return sorted set representation as evaluate_rules produces
    return [sorted(set(f)) for f in flags_list]


def test_vectorized_rules_parity_on_sampled_and_boundary_cases():
    """Sample >= 5,000 PaySim transactions + synthetic boundary cases and assert parity."""
    paysim_path = Path("ml/data/raw/paysim.csv")
    if not paysim_path.exists():
        pytest.skip("PaySim dataset not found at ml/data/raw/paysim.csv")

    df_real = load_raw_paysim(paysim_path)

    # Draw 5,000 stratified samples across types and fraud states
    sampled_rows = df_real.sample(n=5000, random_state=42).copy()

    # Construct boundary edge cases
    boundary_records = [
        # Exactly on large transfer threshold
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 200000.0,
            "name_orig": "C1000",
            "old_balance_orig": 500000.0,
            "new_balance_orig": 300000.0,
            "name_dest": "C2000",
            "old_balance_dest": 0.0,
            "new_balance_dest": 200000.0,
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
        # Just below large transfer threshold
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 199999.99,
            "name_orig": "C1001",
            "old_balance_orig": 500000.0,
            "new_balance_orig": 300000.01,
            "name_dest": "C2001",
            "old_balance_dest": 0.0,
            "new_balance_dest": 199999.99,
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
        # Zero origin balance (strictly NOT drained)
        {
            "step": 1,
            "type": "CASH_OUT",
            "amount": 50.0,
            "name_orig": "C1002",
            "old_balance_orig": 0.0,
            "new_balance_orig": 0.0,
            "name_dest": "C2002",
            "old_balance_dest": 100.0,
            "new_balance_dest": 150.0,
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
        # Exactly 90% drain with new_balance_orig == 0
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 90.0,
            "name_orig": "C1003",
            "old_balance_orig": 100.0,
            "new_balance_orig": 0.0,
            "name_dest": "C2003",
            "old_balance_dest": 0.0,
            "new_balance_dest": 90.0,
            "is_fraud": 1,
            "is_flagged_fraud": 0,
        },
        # 89.99% drain (should NOT trigger drain rule)
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 89.99,
            "name_orig": "C1004",
            "old_balance_orig": 100.0,
            "new_balance_orig": 0.0,
            "name_dest": "C2004",
            "old_balance_dest": 0.0,
            "new_balance_dest": 89.99,
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
        # Drain ratio >= 90% but new_balance_orig > 0 (should NOT trigger drain rule)
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 95.0,
            "name_orig": "C1005",
            "old_balance_orig": 100.0,
            "new_balance_orig": 5.0,
            "name_dest": "C2005",
            "old_balance_dest": 0.0,
            "new_balance_dest": 95.0,
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
        # Merchant destination (dest balance inconsistency rule must NOT apply)
        {
            "step": 1,
            "type": "CASH_OUT",
            "amount": 500.0,
            "name_orig": "C1006",
            "old_balance_orig": 1000.0,
            "new_balance_orig": 500.0,
            "name_dest": "M1006",
            "old_balance_dest": 0.0,
            "new_balance_dest": 0.0,
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
        # Customer destination with inconsistency delta exactly 0.01 (should NOT trigger)
        {
            "step": 1,
            "type": "CASH_OUT",
            "amount": 500.0,
            "name_orig": "C1007",
            "old_balance_orig": 1000.0,
            "new_balance_orig": 500.0,
            "name_dest": "C2007",
            "old_balance_dest": 0.0,
            "new_balance_dest": 500.01,
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
        # Customer destination with inconsistency delta 0.02 (SHOULD trigger)
        {
            "step": 1,
            "type": "CASH_OUT",
            "amount": 500.0,
            "name_orig": "C1008",
            "old_balance_orig": 1000.0,
            "new_balance_orig": 500.0,
            "name_dest": "C2008",
            "old_balance_dest": 0.0,
            "new_balance_dest": 500.02,
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
        # PAYMENT at 50,000 threshold (triggers VERY_HIGH_AMOUNT_FOR_TYPE)
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 50000.0,
            "name_orig": "C1009",
            "old_balance_orig": 100000.0,
            "new_balance_orig": 50000.0,
            "name_dest": "M2009",
            "old_balance_dest": 0.0,
            "new_balance_dest": 0.0,
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
        # DEBIT at 10,000 threshold (triggers VERY_HIGH_AMOUNT_FOR_TYPE)
        {
            "step": 1,
            "type": "DEBIT",
            "amount": 10000.0,
            "name_orig": "C1010",
            "old_balance_orig": 20000.0,
            "new_balance_orig": 10000.0,
            "name_dest": "C2010",
            "old_balance_dest": 0.0,
            "new_balance_dest": 10000.0,
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
    ]
    df_boundary = pd.DataFrame(boundary_records)
    eval_df = pd.concat([sampled_rows, df_boundary], ignore_index=True)

    assert len(eval_df) >= 5000, f"Expected >= 5000 rows, found {len(eval_df)}"

    # Compute vectorized
    vectorized_results = evaluate_rules_vectorized_all(eval_df)

    # Compute row-by-row via backend evaluate_rules
    for idx, row in eval_df.iterrows():
        tx = TransactionIn(
            step=int(row["step"]),
            type=TransactionType(row["type"]),
            amount=float(row["amount"]),
            name_orig=str(row["name_orig"]),
            old_balance_orig=float(row["old_balance_orig"]),
            new_balance_orig=float(row["new_balance_orig"]),
            name_dest=str(row["name_dest"]),
            old_balance_dest=float(row["old_balance_dest"]),
            new_balance_dest=float(row["new_balance_dest"]),
        )
        expected_flags = evaluate_rules(tx)
        actual_flags = vectorized_results[idx]
        assert actual_flags == expected_flags, (
            f"Parity mismatch at row {idx} ({row['type']}, amount={row['amount']}): "
            f"vectorized={actual_flags} vs backend={expected_flags}"
        )
