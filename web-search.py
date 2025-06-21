
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
    return "YAY!!"


def get_openai_response(prompt_text, product_name_for_context, price_cap_for_context):
    """
    Sends a prompt to the OpenAI API and returns the AI's response,
    handling both text and tool calls.
    """
    global current_search_price_cap, current_search_product_name
    current_search_price_cap = price_cap_for_context
    current_search_product_name = product_name_for_context

    messages_for_api = list(conversation_history)

    # Add the user's latest prompt
    messages_for_api.append({"role": "user", "content": prompt_text})

    # Add a system message to guide the AI for the current search (important for tool use decision)
    messages_for_api.append({
        "role": "system",
        "content": (f"The user is searching for '{current_search_product_name}' at or below ${current_search_price_cap:.2f}. "
                    "If you find a suitable product listing, use the 'record_product_found' tool with the exact product name, its price, and the URL. "
                    "Only call the tool if the price is at or below the specified cap. Otherwise, respond in natural language.")
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

        # --- Parse the API's return ---
        if ai_message.tool_calls:
            # The AI wants to call a tool!
            for tool_call in ai_message.tool_calls:
                function_name = tool_call.function.name
                function_args_str = tool_call.function.arguments # This is a JSON string

                print(f"\nAI requested tool call: {function_name} with arguments: {function_args_str}")

                if function_name == "record_product_found":
                    try:
                        args = json.loads(function_args_str) # Parse the JSON string into a Python dict
                        product_name = args.get("product_name")
                        price = args.get("price")
                        url = args.get("url")

                        # double check the price match
                        if price is not None and price <= current_search_price_cap:
                            # Execute your actual "next step" logic
                            tool_output = handle_price_match(product_name, price, url)
                            break
                        else:
                            tool_output = f"\nProduct '{product_name}' found at ${price}, but it's above the cap of ${current_search_price_cap}. Continuing search."
                            print(tool_output)

                        # Add the tool's output back to conversation history so the AI knows the result
                        conversation_history.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_output,
                        })

                        # Make another API call to let the AI summarize/respond after the tool execution
                        # This is important for multi-turn conversations
                        final_response = openai.chat.completions.create(
                            model=MODEL_NAME,
                            messages=conversation_history, # Send updated history with tool output
                        )
                        final_ai_message_content = final_response.choices[0].message.content
                        conversation_history.append({"role": "assistant", "content": final_ai_message_content})
                        return final_ai_message_content

                    except json.JSONDecodeError:
                        print(f"Error parsing JSON arguments for {function_name}: {function_args_str}")
                        return "AI attempted to call tool, but arguments were malformed."
                    except Exception as ex:
                        print(f"Error executing tool {function_name}: {ex}")
                        return f"An error occurred while processing the AI's tool request: {ex}"
                else:
                    return f"AI requested an unknown tool: {function_name}"
        else:
            # The AI returned a regular text message
            ai_message_content = ai_message.content
            conversation_history.append({"role": "assistant", "content": ai_message_content})
            return ai_message_content

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return "Sorry, I encountered an error communicating with OpenAI."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred. Please try again."


def product_web_search_entry(product_name, price_cap_dollors, duration_days):
    print(f"Starting product search for '{product_name}' at or below ${price_cap_dollors:.2f} for {duration_days} days.")
    days_passed = 0

    while days_passed < duration_days:
        delay_seconds = 24 * 60 * 60 # 24 hours

        prompt = f"Please search the web for a product listing that matches '{product_name}' at or below ${price_cap_dollors:.2f}."

        # Pass product_name and price_cap_dollors to get_openai_response for tool context
        ai_response = get_openai_response(prompt, product_name, price_cap_dollors)

        print(f"Search cycle {days_passed + 1}/{duration_days} completed. AI says: {ai_response}")

        days_passed += 1

        if days_passed < duration_days:
            print(f"Waiting for 24 hours before next search cycle. ({duration_days - days_passed} days remaining)")
            time.sleep(delay_seconds)
        else:
            print("Search duration completed.")


# test call
product_web_search_entry("airpods 2nd gen", 500, 28)


# for API-key testing
def chat():
    while True:
        user_input = input("You: ")

        ai_response = get_openai_response(user_input)
        print(f"AI: {ai_response}")

        # exit
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

