"""Pure heuristic safety rules tailored for PaySim financial transaction fraud detection.

All rule thresholds are provisional and subject to empirical calibration by the data
engineering teammate on the training split.
"""

from app.core.config import Settings, settings
from app.models.schemas import TransactionIn, TransactionType


def check_origin_account_drained(
    tx: TransactionIn,
    drain_ratio: float = settings.RULE_DRAIN_BALANCE_RATIO,
) -> str | None:
    """Check if origin account is virtually or completely drained in a single transfer/cash-out."""
    if tx.type not in [TransactionType.TRANSFER, TransactionType.CASH_OUT]:
        return None
    if (
        tx.old_balance_orig > 0.0
        and tx.new_balance_orig == 0.0
        and tx.amount >= tx.old_balance_orig * drain_ratio
    ):
        return "ORIGIN_ACCOUNT_DRAINED"
    return None


def check_large_transfer_amount(
    tx: TransactionIn,
    large_transfer_threshold: float = settings.RULE_LARGE_TRANSFER_THRESHOLD,
) -> str | None:
    """Check if transaction amount equals or exceeds the high-value transfer safeguard threshold."""
    if tx.type not in [TransactionType.TRANSFER, TransactionType.CASH_OUT]:
        return None
    if tx.amount >= large_transfer_threshold:
        return "LARGE_TRANSFER_AMOUNT"
    return None


def check_dest_balance_inconsistency(
    tx: TransactionIn,
) -> str | None:
    """Check if destination balance change fails to reflect transaction amount.

    Scoped strictly to TRANSFER and CASH_OUT, and evaluated only for customer-to-customer
    transactions where destination balance is tracked (i.e. name_dest begins with 'C').
    Merchant destinations ('M...') in PaySim do not track balance deltas.
    """
    if tx.type not in [TransactionType.TRANSFER, TransactionType.CASH_OUT]:
        return None
    if not tx.name_dest.startswith("C"):
        return None

    expected_delta = tx.amount
    actual_delta = tx.new_balance_dest - tx.old_balance_dest
    if abs(actual_delta - expected_delta) > 0.01:
        return "DEST_BALANCE_INCONSISTENCY"
    return None


def check_very_high_amount_for_type(
    tx: TransactionIn,
    high_amount_by_type: dict[str, float] | None = None,
) -> str | None:
    """Check if transaction amount exceeds historical high-percentile threshold for its type."""
    type_thresholds = high_amount_by_type or settings.RULE_HIGH_AMOUNT_BY_TYPE
    threshold = type_thresholds.get(tx.type.value)
    if threshold is not None and tx.amount >= threshold:
        return "VERY_HIGH_AMOUNT_FOR_TYPE"
    return None


def evaluate_rules(
    tx: TransactionIn,
    cfg: Settings | None = None,
) -> list[str]:
    """Execute all heuristic checks on a transaction and return list of triggered flag names."""
    c = cfg or settings
    flags: list[str] = []

    r1 = check_origin_account_drained(tx, drain_ratio=c.RULE_DRAIN_BALANCE_RATIO)
    if r1:
        flags.append(r1)

    r2 = check_large_transfer_amount(
        tx, large_transfer_threshold=c.RULE_LARGE_TRANSFER_THRESHOLD
    )
    if r2:
        flags.append(r2)

    r3 = check_dest_balance_inconsistency(tx)
    if r3:
        flags.append(r3)

    r4 = check_very_high_amount_for_type(
        tx, high_amount_by_type=c.RULE_HIGH_AMOUNT_BY_TYPE
    )
    if r4:
        flags.append(r4)

    return sorted(set(flags))
