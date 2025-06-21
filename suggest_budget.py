#!/usr/bin/env python
import argparse
import json
import os
import sys

from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

def load_financials(path: str) -> list[dict]:
    df = pd.read_excel(path)
    return df.to_dict(orient="records")

def lookup_max_price_via_llm(client: OpenAI, product: str) -> float | None:
    """
    Ask OpenAI for the current average retail price of `product`.
    We prompt it to respond with JUST a number (no dollar sign).
    """
    system = (
        "You are a retail‐pricing assistant. "
        "When asked for the current average retail price of a product, "
        "respond with exactly one number (no currency symbols) "
        "representing the price in USD."
    )
    user = f"What is the current average retail price of an {product} in USD?"
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    text = resp.choices[0].message.content.strip()
    try:
        return float(text.replace(",", ""))
    except ValueError:
        print(f"⚠️ could not parse price from LLM: {text}", file=sys.stderr)
        return None

def build_messages(product: str, timeframe: str, finances: list[dict], max_price: float | None) -> list[dict]:
    cap = (
        f" Do not recommend more than ${max_price:,.2f}, the product’s typical retail cost."
        if max_price is not None
        else ""
    )
    system_prompt = (
        "You are a financial‐planning assistant.\n"
        "Given the user’s last three months of income and expenses and their purchase goal, "
        "recommend a single dollar amount they can comfortably afford. "
        "Respond with exactly one number formatted as a dollar value (e.g. \"$850\"), "
        "with no additional text or explanation."
        + cap
    )
    user_prompt = (
        f"My goal: buy an {product} by {timeframe}.\n"
        "Here are my last three months of finances:\n"
        f"{json.dumps(finances)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

def parse_args():
    p = argparse.ArgumentParser(
        description="Suggest a budget based on your financial history and purchase goal."
    )
    p.add_argument("-f", "--financials", required=True,
        help="Path to your financial .xlsx (last 3 months of rows)")
    p.add_argument("-p", "--product", required=True,
        help="Name of the product you want to buy")
    p.add_argument("-t", "--timeframe", required=True,
        help="By when you want to buy (e.g. '2025-10-21' or 'in 3 months')")
    return p.parse_args()

def main():
    load_dotenv()
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    # 1. Load finances
    try:
        finances = load_financials(args.financials)
    except Exception as e:
        print(f"ERROR reading financials: {e}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()

    # 2. Lookup market price via LLM
    print(f"🔎 Asking LLM for price of “{args.product}”…", file=sys.stderr)
    max_price = lookup_max_price_via_llm(client, args.product)
    if max_price is None:
        print("⚠️ proceeding without a hard cap", file=sys.stderr)
    else:
        print(f"✅ LLM suggests typical price: ${max_price:,.2f}", file=sys.stderr)

    # 3. Build messages & call LLM to suggest budget
    messages = build_messages(args.product, args.timeframe, finances, max_price)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )

    # 4. Parse and clamp
    raw = resp.choices[0].message.content.strip()
    try:
        val = float(raw.replace("$", "").replace(",", ""))
    except ValueError:
        print(f"⚠️ Unexpected response: {raw}", file=sys.stderr)
        sys.exit(1)

    if max_price is not None and val > max_price:
        val = max_price

    # 5. Print final result
    print(f"${val:,.2f}")

if __name__ == "__main__":
    main()
