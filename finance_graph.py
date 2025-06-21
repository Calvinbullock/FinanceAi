from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional, TypedDict
import os
import sys
import json
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from web_search import product_web_search_entry
import datetime


# --- State schema for LangGraph ---
class FinanceState(TypedDict, total=False):
    product: str
    timeframe: str
    financials_path: Optional[str]
    suggested_budget: float
    search_result: Any
    notification_sent: bool

# --- Utility functions from suggest_budget.py ---

def load_financials(path: str) -> list[dict]:
    df = pd.read_excel(path)
    return df.to_dict(orient="records")

def lookup_max_price_via_llm(client: OpenAI, product: str) -> float | None:
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

# --- Agent Node 1: SuggestBudgetAgent ---
def suggest_budget_agent(inputs: FinanceState) -> FinanceState:
    load_dotenv()
    product = inputs["product"]
    timeframe = inputs["timeframe"]
    financials_path = inputs.get("financials_path")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    # 1. Load finances
    finances = []
    if financials_path:
        try:
            finances = load_financials(financials_path)
        except Exception as e:
            raise RuntimeError(f"ERROR reading financials: {e}")

    client = OpenAI()

    # 2. Lookup market price via LLM
    max_price = lookup_max_price_via_llm(client, product)

    # 3. Build messages & call LLM to suggest budget
    messages = build_messages(product, timeframe, finances, max_price)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )

    # 4. Parse and clamp
    raw = resp.choices[0].message.content.strip()
    try:
        val = float(raw.replace("$", "").replace(",", ""))
    except ValueError:
        raise RuntimeError(f"⚠️ Unexpected response: {raw}")

    if max_price is not None and val > max_price:
        val = max_price

    return {
        **inputs,
        "suggested_budget": val,
    }

# --- Agent Node 2: WebSearchAgent ---
def web_search_agent(inputs: FinanceState) -> FinanceState:
    # Import the entry point from web-search.py
    product = inputs["product"]
    price_cap = inputs["suggested_budget"]
    timeframe = inputs.get("timeframe")

    # Compute duration_days from timeframe (assume format "YYYY-MM-DD")
    try:
        if timeframe:
            target_date = datetime.datetime.strptime(timeframe, "%Y-%m-%d")
            today = datetime.datetime.now()
            duration_days = max(1, (target_date - today).days)
        else:
            duration_days = 1
    except Exception as e:
        duration_days = 1  # fallback to 1 day if parsing fails

    # Call the web search entry point
    try:
        result = product_web_search_entry(product, price_cap, duration_days)
    except Exception as e:
        result = f"Error during product web search: {e}"

    return {
        **inputs,
        "search_result": result
    }

# --- Agent Node 3: NotifyAgent ---
def notify_agent(inputs: FinanceState) -> FinanceState:
    # Import send_notification from notify.py
    from notify import send_notification
    load_dotenv()
    product = inputs.get("product")
    search_result = inputs.get("search_result")
    user_contact = None  # Could be extended to take from user input

    # Try to extract product, price, and url(s) from search_result
    if isinstance(search_result, dict):
        product_name = search_result.get("product_name", product)
        price = search_result.get("price")
        url = search_result.get("url")
        links = [url] if url else []
        message = search_result.get("message")
    else:
        # If search_result is a string or unexpected, fallback
        product_name = product
        price = None
        links = []
        message = str(search_result)

    # Use the suggested budget as price if price is missing
    if price is None:
        price = inputs.get("suggested_budget", 0.0)

    # Use API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    # Only send notification if we have a product and price
    sent = False
    if product_name and price:
        sent = send_notification(
            product=product_name,
            price=price,
            links=links,
            user_contact=user_contact,
            api_key=api_key,
            use_email=False  # Set to True if email is configured
        )
    return {
        **inputs,
        "notification_sent": sent
    }

# --- Build the LangGraph DAG ---
def build_finance_graph():
    sg = StateGraph(FinanceState)
    sg.add_node("SuggestBudgetAgent", suggest_budget_agent)
    sg.add_node("WebSearchAgent", web_search_agent)
    sg.add_node("NotifyAgent", notify_agent)
    sg.set_entry_point("SuggestBudgetAgent")
    sg.add_edge("SuggestBudgetAgent", "WebSearchAgent")
    sg.add_edge("WebSearchAgent", "NotifyAgent")
    sg.add_edge("NotifyAgent", END)
    return sg.compile()

# --- Run the DAG ---
import numpy as np
import datetime

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, "isoformat"):
        # Handles datetime, pandas.Timestamp, etc.
        return obj.isoformat()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

def run_finance_graph(product: str, timeframe: str, financials_path: Optional[str] = None):
    graph = build_finance_graph()
    inputs: FinanceState = {
        "product": product,
        "timeframe": timeframe,
        "financials_path": financials_path,
    }
    result = graph.invoke(inputs)
    return make_json_serializable(result)

# Example usage (for testing):
if __name__ == "__main__":
    out = run_finance_graph("airpods 2nd gen", "2025-10-21", "data/bank.xlsx")
    print(out)
