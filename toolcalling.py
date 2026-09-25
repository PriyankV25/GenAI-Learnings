# from dotenv import load_dotenv
# load_dotenv()
# import os

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# import sys
# sys.stdout.reconfigure(encoding="utf-8")
# from langchain.tools import tool

# from rich import print

# #1 creating a tool

# @tool
# def get_text_length(text: str) -> int:
#     """
#     Returns the number of character in a given text
#     """
#     return len(text)



# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1-0528",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
# )



# model = ChatHuggingFace(llm=llm)

# #tool binding
# llm_with_tool = model.bind_tools([get_text_length])

# result = llm_with_tool.invoke("Return the number of character in a given text:hello how are you ")
# print(result.tool_calls)
from dotenv import load_dotenv
load_dotenv()

import os
import sys

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage
from rich import print

sys.stdout.reconfigure(encoding="utf-8")


# -----------------------------
# 1. Define Python Tool
# -----------------------------

@tool
def get_text_length(text: str) -> int:
    """
    Returns the number of characters in a given text.
    """
    return len(text)


# -----------------------------
# 2. Create LLM
# -----------------------------

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

# Give tool information to the LLM
llm_with_tool = model.bind_tools([get_text_length])


# -----------------------------
# 3. User message
# -----------------------------

messages = [
    HumanMessage(
        content="Return the number of characters in this text: hello how are you"
    )
]


# -----------------------------
# 4. Ask LLM
# -----------------------------

result = llm_with_tool.invoke(messages)

print("\n--- LLM RESPONSE ---")
print(result.content)

print("\n--- STRUCTURED TOOL CALL ---")
print(result.tool_calls)


# -----------------------------
# 5. Execute the requested tool
# -----------------------------

if result.tool_calls:

    messages.append(result)

    for tool_call in result.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("\n--- TOOL REQUEST ---")
        print(f"Tool: {tool_name}")
        print(f"Arguments: {tool_args}")

        # Execute the correct tool
        if tool_name == "get_text_length":

            tool_result = get_text_length.invoke(tool_args)

            print("\n--- TOOL EXECUTION RESULT ---")
            print(tool_result)

            # Send result back to LLM
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id
                )
            )


    # -----------------------------
    # 6. Ask LLM for final answer
    # -----------------------------

    final_result = model.invoke(messages)

    print("\n--- FINAL LLM RESPONSE ---")
    print(final_result.content)

else:

    print("\nLLM did not request a tool.")