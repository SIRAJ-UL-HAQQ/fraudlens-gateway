#!/usr/bin/env python3
"""Seed Demo Script for FraudLens Gateway.

Posts varied transactions to /predict via the HTTP API using analyst credentials.
Requires ANALYST_PASSWORD in the environment (no CLI flag for security).
Requires --yes to execute because it inserts permanent audit rows into the database.
"""

import argparse
import os
import random
import sys
from collections import Counter
from typing import Any

import httpx

RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed varied transactions into FraudLens Gateway via REST API."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("GATEWAY_URL", "http://localhost:8000"),
        help="Base URL of FraudLens Gateway (default: GATEWAY_URL env or http://localhost:8000)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("ANALYST_USERNAME"),
        help="Analyst username (default: ANALYST_USERNAME env)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Total number of varied transactions to generate and post (default: 50)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm execution. Required to write permanent records to target gateway.",
    )
    return parser.parse_args()


def generate_varied_transactions(n: int) -> list[dict[str, Any]]:
    """Generate deterministic varied PaySim transactions (fraud, legit, zero, borderline)."""
    random.seed(RANDOM_SEED)
    transactions: list[dict[str, Any]] = []

    for i in range(1, n + 1):
        step = (i % 24) + 1
        scenario = i % 4

        if scenario == 0:
            # Obvious Fraud: high amount TRANSFER draining origin account completely
            amount = round(random.uniform(200_000.0, 1_500_000.0), 2)
            tx = {
                "step": step,
                "type": "TRANSFER",
                "amount": amount,
                "name_orig": f"C_FRAUD_{i:04d}",
                "old_balance_orig": amount,
                "new_balance_orig": 0.0,
                "name_dest": f"C_MULE_{i:04d}",
                "old_balance_dest": 0.0,
                "new_balance_dest": 0.0,
            }
        elif scenario == 1:
            # Obvious Legitimate: normal PAYMENT with consistent small balance delta
            amount = round(random.uniform(5.0, 250.0), 2)
            old_bal = round(amount + random.uniform(500.0, 5000.0), 2)
            tx = {
                "step": step,
                "type": "PAYMENT",
                "amount": amount,
                "name_orig": f"C_LEGIT_{i:04d}",
                "old_balance_orig": old_bal,
                "new_balance_orig": round(old_bal - amount, 2),
                "name_dest": f"M_MERCH_{i:04d}",
                "old_balance_dest": 0.0,
                "new_balance_dest": 0.0,
            }
        elif scenario == 2:
            # Zero-amount edge case
            tx = {
                "step": step,
                "type": "TRANSFER" if i % 2 == 0 else "CASH_OUT",
                "amount": 0.0,
                "name_orig": f"C_ZERO_{i:04d}",
                "old_balance_orig": 1000.0,
                "new_balance_orig": 1000.0,
                "name_dest": f"C_DEST_{i:04d}",
                "old_balance_dest": 500.0,
                "new_balance_dest": 500.0,
            }
        else:
            # Borderline / Analyst Review case: high amount without draining origin
            amount = round(random.uniform(60_000.0, 150_000.0), 2)
            old_bal = round(amount * 4.0, 2)
            tx = {
                "step": step,
                "type": "CASH_OUT",
                "amount": amount,
                "name_orig": f"C_BORDER_{i:04d}",
                "old_balance_orig": old_bal,
                "new_balance_orig": round(old_bal - amount, 2),
                "name_dest": f"M_DEST_{i:04d}",
                "old_balance_dest": 1000.0,
                "new_balance_dest": round(1000.0 + amount, 2),
            }

        transactions.append(tx)

    return transactions


def login_analyst(client: httpx.Client, base_url: str, username: str, password: str) -> str:
    """Authenticate against /api/v1/auth/login and return access token."""
    login_url = f"{base_url.rstrip('/')}/api/v1/auth/login"
    resp = client.post(login_url, json={"username": username, "password": password}, timeout=10.0)
    if resp.status_code != 200:
        print(f"Authentication failed ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    token = data.get("access_token")
    if not token:
        print("Login response did not contain 'access_token'.", file=sys.stderr)
        sys.exit(1)
    return token


def main() -> None:
    args = parse_args()
    base_url = args.url.rstrip("/")

    # Security: Password must come from environment only
    password = os.environ.get("ANALYST_PASSWORD")
    if not password:
        print(
            "ERROR: Missing required environment variable ANALYST_PASSWORD.\n"
            "Passwords must never be passed as CLI arguments.",
            file=sys.stderr,
        )
        sys.exit(1)

    username = args.username
    if not username:
        print(
            "ERROR: Missing username. Supply --username or set ANALYST_USERNAME environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 65)
    print("FraudLens Gateway — Transaction Seeder")
    print(f"Target Gateway URL : {base_url}")
    print(f"Authenticated User : {username}")
    print(f"Transactions Count : {args.count}")
    print("=" * 65)
    print("WARNING: This script writes permanent records to the database.")
    print("Ensure this command is directed at a throwaway or dev environment.")

    if not args.yes:
        print(
            "\nAborted: --yes was not provided. Run with --yes to confirm execution.\n"
            "Example: python scripts/seed_demo.py --yes",
            file=sys.stderr,
        )
        sys.exit(1)

    transactions = generate_varied_transactions(args.count)

    with httpx.Client(timeout=15.0) as client:
        # Initial login
        token = login_analyst(client, base_url, username, password)
        predict_url = f"{base_url}/api/v1/predict"

        decision_counts: Counter[str] = Counter()
        stability_counts: Counter[str] = Counter()
        successful_posts = 0

        print(f"\nPosting {len(transactions)} transactions...")
        for idx, tx in enumerate(transactions, 1):
            headers = {"Authorization": f"Bearer {token}"}
            resp = client.post(predict_url, json=tx, headers=headers)

            # Handle 401: one re-login retry
            if resp.status_code == 401:
                print(f"[{idx}] Received 401 Unauthorized. Re-authenticating...")
                token = login_analyst(client, base_url, username, password)
                headers = {"Authorization": f"Bearer {token}"}
                resp = client.post(predict_url, json=tx, headers=headers)

            if resp.status_code == 200:
                result = resp.json()
                decision = result.get("gateway_decision", "Unknown")
                stability = result.get("explanation_status", "Unknown")
                decision_counts[decision] += 1
                stability_counts[stability] += 1
                successful_posts += 1
            else:
                print(f"[{idx}] Failed to post transaction ({resp.status_code}): {resp.text}", file=sys.stderr)

        print("\n" + "=" * 45)
        print("Seeding Run Summary:")
        print(f"Total Attempted   : {len(transactions)}")
        print(f"Successfully Scored: {successful_posts}")
        print("\nDecision Breakdown:")
        for dec, cnt in decision_counts.items():
            print(f"  - {dec:<16}: {cnt}")
        print("\nExplanation Stability Breakdown:")
        for stab, cnt in stability_counts.items():
            print(f"  - {stab:<16}: {cnt}")
        print("=" * 45)


if __name__ == "__main__":
    main()
