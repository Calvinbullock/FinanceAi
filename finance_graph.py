from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional, TypedDict
import os
import sys
import json
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

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
    load_dotenv()
    product = inputs["product"]
    price_cap = inputs["suggested_budget"]
    timeframe = inputs["timeframe"]

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_product_found",
                "description": "Records details of a product found during a web search that meets the specified criteria.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_name": {
                            "type": "string",
                            "description": "The exact name of the product found, e.g., 'Sony WH-1000XM5 headphones'."
                        },
                        "price": {
                            "type": "number",
                            "description": "The numeric price of the product found, e.g., 299.99."
                        },
                        "url": {
                            "type": "string",
                            "description": "The URL of the product listing."
                        }
                    },
                    "required": ["product_name", "price", "url"]
                }
            }
        }
    ]

    prompt = f"Please search the web for a product listing that matches '{product}' at or below ${price_cap:.2f}."

    system_message = {
        "role": "system",
        "content": (
            f"The user is searching for '{product}' at or below ${price_cap:.2f}. "
            "If you find a suitable product listing, use the 'record_product_found' tool with the exact product name, its price, and the URL. "
            "Only call the tool if the price is at or below the specified cap. Otherwise, respond in natural language."
        )
    }

    messages = [
        {"role": "user", "content": prompt},
        system_message
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=250,
            temperature=0.7,
            stream=False
        )

        ai_message = response.choices[0].message

        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            for tool_call in ai_message.tool_calls:
                function_name = tool_call.function.name
                function_args_str = tool_call.function.arguments

                if function_name == "record_product_found":
                    try:
                        args = json.loads(function_args_str)
                        product_name = args.get("product_name")
                        price = args.get("price")
                        url = args.get("url")
                        if price is not None and price <= price_cap:
                            return {
                                **inputs,
                                "search_result": {
                                    "product_name": product_name,
                                    "price": price,
                                    "url": url,
                                    "message": f"Found {product_name} for ${price} at {url}"
                                }
                            }
                        else:
                            return {
                                **inputs,
                                "search_result": f"Product '{product_name}' found at ${price}, but it's above the cap of ${price_cap:.2f}."
                            }
                    except Exception as ex:
                        return {
                            **inputs,
                            "search_result": f"Error parsing tool call arguments: {ex}"
                        }
                else:
                    return {
                        **inputs,
                        "search_result": f"AI requested an unknown tool: {function_name}"
                    }
        else:
            return {
                **inputs,
                "search_result": ai_message.content
            }

    except Exception as e:
        return {
            **inputs,
            "search_result": f"Error during OpenAI web search: {e}"
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
