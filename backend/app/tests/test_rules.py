from app.models.schemas import TransactionIn, TransactionType
from app.services.rules import (
    check_dest_balance_inconsistency,
    check_large_transfer_amount,
    check_origin_account_drained,
    check_very_high_amount_for_type,
    evaluate_rules,
)


def make_tx(**kwargs) -> TransactionIn:
    defaults = {
        "step": 1,
        "type": TransactionType.TRANSFER,
        "amount": 500.0,
        "name_orig": "C100000001",
        "old_balance_orig": 1000.0,
        "new_balance_orig": 500.0,
        "name_dest": "C200000002",
        "old_balance_dest": 0.0,
        "new_balance_dest": 500.0,
    }
    defaults.update(kwargs)
    return TransactionIn(**defaults)


def test_origin_account_drained_boundary():
    # Exactly 90% with new_balance 0 -> triggers
    tx_exact_90 = make_tx(old_balance_orig=100.0, new_balance_orig=0.0, amount=90.0)
    assert check_origin_account_drained(tx_exact_90) == "ORIGIN_ACCOUNT_DRAINED"

    # Just below 90% -> does not trigger
    tx_below_90 = make_tx(old_balance_orig=100.0, new_balance_orig=0.0, amount=89.99)
    assert check_origin_account_drained(tx_below_90) is None

    # New balance > 0 -> does not trigger
    tx_nonzero_new = make_tx(
        old_balance_orig=100.0, new_balance_orig=0.01, amount=99.99
    )
    assert check_origin_account_drained(tx_nonzero_new) is None

    # Old balance zero -> does not trigger
    tx_zero_old = make_tx(old_balance_orig=0.0, new_balance_orig=0.0, amount=50.0)
    assert check_origin_account_drained(tx_zero_old) is None

    # Non-transfer/cash_out (e.g. PAYMENT) -> does not trigger
    tx_payment = make_tx(
        type=TransactionType.PAYMENT,
        old_balance_orig=100.0,
        new_balance_orig=0.0,
        amount=100.0,
    )
    assert check_origin_account_drained(tx_payment) is None


def test_large_transfer_amount_boundary():
    # Exactly 200k on TRANSFER -> triggers
    tx_exact = make_tx(amount=200_000.0)
    assert check_large_transfer_amount(tx_exact) == "LARGE_TRANSFER_AMOUNT"

    # 199,999.99 -> does not trigger
    tx_below = make_tx(amount=199_999.99)
    assert check_large_transfer_amount(tx_below) is None

    # Exactly 200k on CASH_OUT -> triggers
    tx_cashout = make_tx(type=TransactionType.CASH_OUT, amount=200_000.0)
    assert check_large_transfer_amount(tx_cashout) == "LARGE_TRANSFER_AMOUNT"

    # 200k on PAYMENT -> does not trigger (scoped to TRANSFER/CASH_OUT)
    tx_payment = make_tx(type=TransactionType.PAYMENT, amount=200_000.0)
    assert check_large_transfer_amount(tx_payment) is None


def test_dest_balance_inconsistency():
    # Consistent: old=0, new=500, amount=500 -> None
    tx_consistent = make_tx(
        name_dest="C12345",
        old_balance_dest=0.0,
        new_balance_dest=500.0,
        amount=500.0,
    )
    assert check_dest_balance_inconsistency(tx_consistent) is None

    # Inconsistent: old=0, new=0, amount=500 -> triggers
    tx_inconsistent = make_tx(
        name_dest="C12345",
        old_balance_dest=0.0,
        new_balance_dest=0.0,
        amount=500.0,
    )
    assert (
        check_dest_balance_inconsistency(tx_inconsistent)
        == "DEST_BALANCE_INCONSISTENCY"
    )

    # Merchant destination ('M...') ignored even if balance delta doesn't match
    tx_merchant = make_tx(
        name_dest="M98765",
        old_balance_dest=0.0,
        new_balance_dest=0.0,
        amount=500.0,
    )
    assert check_dest_balance_inconsistency(tx_merchant) is None

    # PAYMENT type ignored (scoped strictly to TRANSFER and CASH_OUT)
    tx_payment = make_tx(
        type=TransactionType.PAYMENT,
        name_dest="C12345",
        old_balance_dest=0.0,
        new_balance_dest=0.0,
        amount=500.0,
    )
    assert check_dest_balance_inconsistency(tx_payment) is None


def test_very_high_amount_for_type():
    # PAYMENT threshold = 50,000.0
    tx_pay_high = make_tx(type=TransactionType.PAYMENT, amount=50_000.0)
    assert check_very_high_amount_for_type(tx_pay_high) == "VERY_HIGH_AMOUNT_FOR_TYPE"

    tx_pay_below = make_tx(type=TransactionType.PAYMENT, amount=49_999.99)
    assert check_very_high_amount_for_type(tx_pay_below) is None

    # DEBIT threshold = 10,000.0
    tx_debit = make_tx(type=TransactionType.DEBIT, amount=10_000.0)
    assert check_very_high_amount_for_type(tx_debit) == "VERY_HIGH_AMOUNT_FOR_TYPE"


def test_evaluate_rules_multiple_flags():
    # Origin drained + Large transfer + Dest inconsistency
    tx_multi = make_tx(
        type=TransactionType.TRANSFER,
        amount=250_000.0,
        name_orig="C111",
        old_balance_orig=250_000.0,
        new_balance_orig=0.0,
        name_dest="C222",
        old_balance_dest=0.0,
        new_balance_dest=0.0,
    )
    flags = evaluate_rules(tx_multi)
    assert "ORIGIN_ACCOUNT_DRAINED" in flags
    assert "LARGE_TRANSFER_AMOUNT" in flags
    assert "DEST_BALANCE_INCONSISTENCY" in flags
    assert len(flags) >= 3
