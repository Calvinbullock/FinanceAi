# --- this is the entry point function ---
# product_web_search_entry(product_name, price_cap_dollors, duration_days):

from os import environ
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = environ["OPENAI_API_KEY"]

import openai
import datetime
import time
import json

MODEL_NAME = "gpt-4.1"

# This list will store the conversation history, which is crucial for
# the AI to maintain context in a conversation.
conversation_history = []

# --- Global flag to signal loop termination ---
price_match_found = False

# Global variable to store the current price cap for the specific search run
# This is a simple way to pass context to the tool handler
current_search_price_cap = 0.0
current_search_product_name = ""

# This list describes the functions/tools available to the OpenAI model
tools = [
    {
        "type": "function",
        "function": {
            "name": "record_product_found", # This is the name the AI will "call"
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
                "required": ["product_name", "price", "url"] # These fields MUST be provided by the AI
            }
        }
    }
]


def handle_price_match(product_name: str, price: float, url: str):
    """
    Handles the logic when a product is found that matches the desired price criteria.

    This function serves as the "next step" once the AI successfully identifies a product
    within the specified price range. It prints a success message, sets a global flag
    to indicate a match, and returns a confirmation string.

    Args:
        product_name (str): The name of the product found.
        price (float): The price of the product found.
        url (str): The URL of the product listing.

    Returns:
        str: A message confirming that the price match was handled.
    """
    global price_match_found
    print(f"\n*** PRICE MATCH FOUND! ***")
    print(f"Product: {product_name}")
    print(f"Price: ${price:.2f}")
    print(f"URL: {url}")
    print(f"*** Initiating next step... (e.g., sending alert, recording, purchasing) ***\n")

    price_match_found = True # Set the flag to True to signal termination of the search loop

    return "Price match handled successfully by Python code."


BLOCKED_DOMAINS = []

def is_blocked_url(url):
    return any(domain in url for domain in BLOCKED_DOMAINS)

def get_openai_response(prompt_text, product_name_for_context, price_cap_for_context):
    """
    Sends a prompt to the OpenAI API and processes its response, supporting both
    natural language replies and AI-initiated tool calls.

    This function manages the conversation history, constructs API requests,
    and interprets the AI's response. If the AI decides to use a defined tool
    (like `record_product_found`), it executes the corresponding Python logic
    and feeds the result back to the AI for further context.

    Args:
        prompt_text (str): The natural language prompt or query to send to the AI.
        product_name_for_context (str): The name of the product currently being searched.
                                        Used to provide context to the AI for accurate tool use.
        price_cap_for_context (float): The maximum acceptable price for the product.
                                       Used to provide context to the AI for accurate tool use
                                       and for internal validation of tool call arguments.

    Returns:
        str: The AI's final natural language response after processing the prompt
             and any potential tool calls.

    Raises:
        openai.APIError: If there's an issue communicating with the OpenAI API.
        Exception: For any other unexpected errors during the process.
    """
    global current_search_price_cap, current_search_product_name, price_match_found
    current_search_price_cap = price_cap_for_context
    current_search_product_name = product_name_for_context

    messages_for_api = list(conversation_history)

    # Add the user's latest prompt
    messages_for_api.append({"role": "user", "content": prompt_text})

    # Add a system message to guide the AI for the current search (important for tool use decision)
    messages_for_api.append({
        "role": "system",
        "content": (
            f"The user is searching for '{current_search_product_name}' at or below ${current_search_price_cap:.2f}. "
            "If you find ANY product listing (real or hypothetical) that matches the criteria, you MUST use the 'record_product_found' tool with the exact product name, its price, and the URL. "
            "The URL you provide should be plausible and look like a real product listing from a well-known retailer (such as Amazon, Best Buy, Walmart, Target, etc.). "
            "Do NOT invent random or obviously fake links. If you cannot find any product at or below the price cap, only then respond in natural language."
        )
    })

    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_for_api,
            tools=tools,
            tool_choice="auto",
            max_tokens=250,
            temperature=0.7,
            stream=False
        )

        ai_message = response.choices[0].message
        conversation_history.append(ai_message)

        if ai_message.tool_calls:
            # If the AI responded with tool_calls, process them
            for tool_call in ai_message.tool_calls: # Assuming only one tool call for simplicity in this example
                function_name = tool_call.function.name
                function_args_str = tool_call.function.arguments

                print(f"AI requested tool call: {function_name} with arguments: {function_args_str}")

                if function_name == "record_product_found":
                    try:
                        args = json.loads(function_args_str)
                        product_name = args.get("product_name")
                        price = args.get("price")
                        url = args.get("url")

                        if price is not None and price <= current_search_price_cap and url:
                            if is_blocked_url(url):
                                tool_output = f"Product '{product_name}' found at ${price}, but the URL is blocked ({url}). Skipping."
                                print(tool_output)
                                continue  # Skip this result and continue searching
                            tool_output = handle_price_match(product_name, price, url)
                            price_match_found = True
                            # Return structured result
                            return {
                                "product_name": product_name,
                                "price": price,
                                "url": url,
                                "message": tool_output
                            }
                        else:
                            tool_output = f"Product '{product_name}' found at ${price}, but it's above the cap of ${current_search_price_cap:.2f} or missing URL. Continuing search."
                            print(tool_output)

                        conversation_history.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_output,
                        })

                        final_response_after_tool = openai.chat.completions.create(
                            model=MODEL_NAME,
                            messages=conversation_history, # Now includes the assistant's tool_calls and the tool's output
                        )
                        final_ai_message_content = final_response_after_tool.choices[0].message.content
                        conversation_history.append({"role": "assistant", "content": final_ai_message_content})
                        return {
                            "message": final_ai_message_content
                        }

                    except json.JSONDecodeError:
                        print(f"Error parsing JSON arguments for {function_name}: {function_args_str}")
                        return {"message": "AI attempted to call tool, but arguments were malformed."}
                    except Exception as ex:
                        print(f"Error executing tool {function_name}: {ex}")
                        return {"message": f"An error occurred while processing the AI's tool request: {ex}"}

                else:
                    return {"message": f"AI requested an unknown tool: {function_name}"}
        else:
            # The AI returned a regular text message
            ai_message_content = ai_message.content
            conversation_history.append({"role": "assistant", "content": ai_message_content})
            # Try to extract a URL if present (fallback)
            import re
            url_match = re.search(r'https?://\S+', ai_message_content)
            if url_match:
                url = url_match.group(0)
                if is_blocked_url(url):
                    return {"message": f"Found a link, but it is blocked: {url}"}
                return {
                    "product_name": current_search_product_name,
                    "price": None,
                    "url": url,
                    "message": ai_message_content
                }
            return {"message": ai_message_content}

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return "Sorry, I encountered an error communicating with OpenAI."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred. Please try again."




def product_web_search_once(product_name, price_cap_dollors):
    """
    Performs a single-cycle product web search (no sleep, no loop).
    Returns the result of one call to get_openai_response.
    """
    prompt = f"Please search the web for a product listing that matches '{product_name}' at or below ${price_cap_dollors:.2f}."
    return get_openai_response(prompt, product_name, price_cap_dollors)

#    --- CALL THIS TO USE THIS AGENT ---
# --- this is the entry point function ---
def product_web_search_entry(product_name, price_cap_dollors, duration_days):
    """
    Manages the automated web search for a product over a specified duration.

    This function iteratively prompts the AI to search for a product within a given
    price cap. It continues for a set number of days or until a price match is found.
    It incorporates delays between search cycles.

    Args:
        product_name (str): The name of the product to search for.
        price_cap_dollars (float): The maximum price (in dollars) for the product.
        duration_days (int): The total number of days to continue the search.
                             The search will stop either after `duration_days`
                             or if a price match is found, whichever comes first.
    """
    global price_match_found

    print(f"Starting product search for '{product_name}' at or below ${price_cap_dollors:.2f} for {duration_days} days.")
    days_passed = 0

    while not price_match_found:
        delay_seconds = 24 * 60 * 60 # 24 hours

        prompt = f"Please search the web for a product listing that matches '{product_name}' at or below ${price_cap_dollors:.2f}."

        # Pass product_name and price_cap_dollors to get_openai_response for tool context
        ai_response = get_openai_response(prompt, product_name, price_cap_dollors)

        print(f"Search cycle {days_passed + 1}/{duration_days} completed. AI says: {ai_response}")

        days_passed += 1

        if days_passed < duration_days and not price_match_found:
            print(f"Waiting for 24 hours before next search cycle. ({duration_days - days_passed} days remaining)")
            time.sleep(delay_seconds)
        else:
            print("Search duration completed.")


# test call
# product_web_search_entry("laptop", 1300, 2)
