# app.py
import tempfile
import os
from typing import Optional

import streamlit as st
import pandas as pd

from cli import parse_intent, parse_timeframe_to_date, entry_point, FinanceAI

st.set_page_config(page_title="Finance-AI Dashboard", layout="centered")
st.title("💰 Finance-AI Budget Planner")

def analyze_goal(user_text: str) -> Optional[object]:
    """Parse intent from user text, handling errors."""
    try:
        intent = parse_intent(user_text)
    except Exception as e:
        st.error(f"Failed to parse intent: {e}")
        return None
    return intent

def display_intent(intent) -> None:
    """Display parsed intent parameters in a clean format."""
    st.subheader("🔍 Parsed Parameters")
    st.write(f"- **Product:** {getattr(intent, 'product', '—')}")
    st.write(f"- **By:** {getattr(intent, 'timeframe', '—')}")
    st.write(f"- **Budget (if any):** {getattr(intent, 'desired_price', '—') or '—'}")
    st.write(f"- **Location:** {getattr(intent, 'location', '—') or '—'}")
    st.write(f"- **Upload financials?** {'Yes' if getattr(intent, 'provide_financials', False) else 'No'}")

def handle_file_upload() -> Optional[str]:
    """Handle file upload and return path to temp file, or None."""
    uploaded_file = st.file_uploader("Upload your financial .xlsx", type=["xlsx", "xls"], key="file_uploader")
    if uploaded_file is None:
        return None
    tf = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tf.write(uploaded_file.read())
    tf.flush()
    tf.close()
    return tf.name

def cleanup_temp_file(path: Optional[str]) -> None:
    """Delete temp file if it exists."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            st.warning(f"Could not delete temp file: {e}")

def prompt_missing_fields(intent_data: dict) -> dict:
    """Prompt user for any missing fields and return updated data."""
    # Product
    if not intent_data.get("product"):
        intent_data["product"] = st.text_input("What are you looking to buy?", key="product")
    # Timeframe
    tf = intent_data.get("timeframe", "")
    if not tf or tf.lower() in ("unspecified", "unknown", "n/a"):
        intent_data["timeframe"] = st.text_input("When would you like to buy it by? (e.g. 'in 3 months')", key="timeframe")
    # Budget
    if not intent_data.get("desired_price"):
        intent_data["desired_price"] = st.text_input("What's your budget (or leave blank)?", value="", key="desired_price")
    # Location
    if not intent_data.get("location"):
        intent_data["location"] = st.text_input("Buying online or in-store? Which country/region?", key="location")
    # Financials consent — always ask, even if LLM filled it
    provide_financials = st.radio(
        "Would you like to upload your financial spreadsheet?",
        options=["No", "Yes"],
        index=1 if intent_data.get("provide_financials") else 0,
        key="provide_financials_radio"
    )
    intent_data["provide_financials"] = (provide_financials == "Yes")
    return intent_data

def all_fields_filled(intent_data: dict) -> bool:
    """Check if all required fields are filled."""
    required = ["product", "timeframe", "provide_financials"]
    for field in required:
        if intent_data.get(field) in [None, ""]:
            return False
    return True

def main():
    # Use session state to manage multi-step flow
    if "intent" not in st.session_state:
        st.session_state.intent = None
    if "intent_data" not in st.session_state:
        st.session_state.intent_data = None
    if "xlsx_path" not in st.session_state:
        st.session_state.xlsx_path = None
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    if "dispatched" not in st.session_state:
        st.session_state.dispatched = False
    if "ready_to_confirm" not in st.session_state:
        st.session_state.ready_to_confirm = False
    if "last_user_text" not in st.session_state:
        st.session_state.last_user_text = ""

    user_text = st.text_input(
        "Describe your purchase goal",
        placeholder="e.g. I want to buy an iPhone in the next 3 months"
    )

    # Reset flow if user_text changes
    if user_text != st.session_state.last_user_text:
        st.session_state.analyzed = False
        st.session_state.intent = None
        st.session_state.intent_data = None
        st.session_state.xlsx_path = None
        st.session_state.dispatched = False
        st.session_state.ready_to_confirm = False
        st.session_state.last_user_text = user_text

    if not user_text:
        st.info("Enter what you'd like to purchase and when.")
        st.stop()

    analyze = st.button("Analyze Goal", disabled=st.session_state.analyzed)
    if analyze and not st.session_state.analyzed:
        with st.spinner("Extracting intent…"):
            intent = analyze_goal(user_text)
            if intent is not None:
                st.session_state.intent = intent
                st.session_state.intent_data = intent.model_dump()
                st.session_state.analyzed = True
            else:
                st.stop()

    if st.session_state.analyzed and st.session_state.intent_data and not st.session_state.ready_to_confirm:
        # Prompt for missing fields
        st.session_state.intent_data = prompt_missing_fields(st.session_state.intent_data)
        if all_fields_filled(st.session_state.intent_data):
            if st.button("Continue"):
                st.session_state.ready_to_confirm = True
        else:
            st.info("Please fill in all required fields above to continue.")

    if st.session_state.ready_to_confirm:
        # Build FinanceAI object
        try:
            st.session_state.intent_data["timeframe"] = parse_timeframe_to_date(st.session_state.intent_data["timeframe"])
            intent_obj = FinanceAI(**st.session_state.intent_data)
            display_intent(intent_obj)
        except Exception as e:
            st.error(f"Error constructing intent: {e}")
            st.stop()

        # If financials are required, prompt for upload and require it before proceeding
        xlsx_path = None
        needs_financials = getattr(intent_obj, "provide_financials", False)
        if needs_financials:
            xlsx_path = handle_file_upload()
            if xlsx_path:
                st.session_state.xlsx_path = xlsx_path
            else:
                st.warning("Please upload your .xlsx file to continue.")
        else:
            st.session_state.xlsx_path = None

        # Only enable confirm if either not needed or file uploaded
        can_confirm = (not needs_financials) or (needs_financials and st.session_state.xlsx_path)
        confirm = st.button("Confirm & Run Agents", disabled=st.session_state.dispatched or not can_confirm)
        if confirm and not st.session_state.dispatched and can_confirm:
            with st.spinner("Dispatching agents…"):
                try:
                    result = entry_point(intent_obj, st.session_state.xlsx_path, return_result=True)
                    st.session_state.agent_result = result
                    st.session_state.dispatched = True
                except Exception as e:
                    st.error(f"Agent run failed: {e}")
                    st.stop()
            st.success("✅ Agents have been dispatched. See results below.")
            # Clean up temp file after use
            cleanup_temp_file(st.session_state.xlsx_path)

        # Show agent results in a nice box if available
        if st.session_state.get("agent_result"):
            result = st.session_state.agent_result
            with st.container():
                st.markdown(
                    f"""
                    <div style="border:2px solid #4CAF50; border-radius:10px; padding:20px; background-color:#f9f9f9;">
                        <h4 style="color:#4CAF50; margin-top:0;">Agent Results</h4>
                        <b>Product:</b> {result.get("product", "—")}<br>
                        <b>Timeframe:</b> {result.get("timeframe", "—")}<br>
                        <b>Budget:</b> ${result.get("suggested_budget", "—")}<br>
                        <b>Financials Path:</b> {result.get("financials_path", "—")}<br>
                        <b>Notification Sent:</b> {"✅" if result.get("notification_sent") else "❌"}<br>
                        <hr>
                        <b>Search Result:</b><br>
                        {"<ul><li><b>Name:</b> " + result["search_result"].get("product_name", "—") + "</li>"
                          + "<li><b>Price:</b> $" + str(result["search_result"].get("price", "—")) + "</li>"
                          + "<li><b>URL:</b> <a href='" + result["search_result"].get("url", "#") + "' target='_blank'>" + result["search_result"].get("url", "—") + "</a></li>"
                          + "<li><b>Message:</b> " + result["search_result"].get("message", "—") + "</li></ul>"
                          if isinstance(result.get("search_result"), dict)
                          else str(result.get("search_result", "—"))
                        }
                    </div>
                    """,
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
