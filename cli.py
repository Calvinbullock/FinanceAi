#!/usr/bin/env python
import pathlib
import typer
import json
import dateparser
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from typing import Optional


load_dotenv()
app = typer.Typer()
client = OpenAI()

# 1. Data model
class FinanceAI(BaseModel):
    product: str = Field(..., description="Desired product or service")
    timeframe: str = Field(..., description="When they want to buy")
    desired_price: Optional[str] = Field(None, description="Target or max price")
    location: Optional[str] = Field(None, description="Physical or online region")
    provide_financials: bool = Field(..., description="Whether they will upload data")

# 2. Clean Pydantic schema for OpenAI
def openai_schema(model_cls) -> dict:
    schema = model_cls.model_json_schema()
    for prop in schema.get("properties", {}).values():
        for composite in ("anyOf", "oneOf", "allOf"):
            if composite in prop:
                first_type = next(
                    (sub.get("type") for sub in prop[composite] if sub.get("type") != "null"),
                    "string",
                )
                prop.clear()
                prop["type"] = first_type
        prop.pop("default", None)
        prop.pop("title", None)
    return schema

# 3. Follow-up on missing fields
def ask_missing(intent: FinanceAI) -> FinanceAI:
    data = intent.model_dump()  # use model_dump instead of deprecated .dict()

    # 1⃣ Product & timeframe
    if not data.get("product"):
        data["product"] = typer.prompt("What are you looking to buy?")
    tf = data.get("timeframe", "")
    if not tf or tf.lower() in ("unspecified", "unknown", "n/a"):
        data["timeframe"] = typer.prompt(
            "When would you like to buy it by? (e.g. 'in 3 months')"
        )

    # 2⃣ Budget & location
    if not data.get("desired_price"):
        data["desired_price"] = typer.prompt("What's your budget (or leave blank)?", default="")
    if not data.get("location"):
        data["location"] = typer.prompt(
            "Buying online or in-store? Which country/region?"
        )

    # 3⃣ Financials consent — always confirm
    wants = typer.confirm("Would you like to upload your financial spreadsheet?")
    data["provide_financials"] = wants

    return FinanceAI(**data)

# 4. Get .xlsx path
def get_financial_path() -> pathlib.Path:
    path = typer.prompt("Path to your financial .xlsx")
    p = pathlib.Path(path).expanduser()
    if not (p.exists() and p.suffix in {".xlsx", ".xls"}):
        typer.echo("File not found or wrong format! Try again.")
        raise typer.Exit(code=1)
    return p

# 5. Entry point: call the LangGraph DAG
def entry_point(intent: FinanceAI, xlsx_path: Optional[pathlib.Path]):
    from finance_graph import run_finance_graph

    product = intent.product
    timeframe = intent.timeframe
    financials_path = str(xlsx_path) if xlsx_path else None

    typer.echo("\nRunning agents (SuggestBudget → WebSearch)...")
    result = run_finance_graph(product, timeframe, financials_path)
    typer.echo("\n--- Agent Results ---")
    typer.echo(json.dumps(result, indent=2))

# Parse almost any human-friendly timeframe into YYYY-MM-DD
def parse_timeframe_to_date(tf: str) -> str:
    # Try direct parse with preference for future
    dt = dateparser.parse(
        tf,
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": datetime.now(),
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )
    if dt:
        return dt.date().isoformat()

    # As a final fallback, return the original
    return tf

# 6. Parse with the LLM (synchronously)
def parse_intent(user_text: str) -> FinanceAI:
    schema = openai_schema(FinanceAI)

    chat = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You extract structured purchase intents."},
            {"role": "user",   "content": user_text},
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "extract_intent",
                "description": "Extract key fields about a purchase request",
                "parameters": schema,
            },
        }],
        tool_choice={"type": "function", "function": {"name": "extract_intent"}},
    )

    calls = chat.choices[0].message.tool_calls
    if not calls:
        raise RuntimeError("LLM did not return a function call")

    # Fix: arguments live under .function.arguments
    raw_json = calls[0].function.arguments
    return FinanceAI.model_validate(json.loads(raw_json))

# 7. CLI command
@app.command()
def run():
    # 1. Free-form prompt
    user_text = typer.prompt("Tell me in natural language what you’d like to do")

    # 2. Extract with LLM
    try:
        intent = parse_intent(user_text)
    except (ValidationError, RuntimeError) as e:
        typer.echo(f"Error extracting intent: {e}")
        raise typer.Exit(code=1)

    # 3. Fill missing fields
    intent = ask_missing(intent)
    intent.timeframe = parse_timeframe_to_date(intent.timeframe)

    # 4. Optionally ask for .xlsx
    xlsx_path = None
    if intent.provide_financials:
        xlsx_path = get_financial_path()

    # 5. Summary & confirm
    typer.echo("\n--- Summary ---")
    typer.echo(intent.model_dump_json(indent=2))
    if xlsx_path:
        typer.echo(f"Financials path: {xlsx_path}")
    if not typer.confirm("Proceed with these details?"):
        raise typer.Exit()

    # 6. Hand off
    entry_point(intent, xlsx_path)
    typer.echo("✅ Done – the agents are on it!")

if __name__ == "__main__":
    app()
