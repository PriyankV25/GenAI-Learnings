
from dotenv import load_dotenv

load_dotenv()

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq",
    temperature=0.9
)

messages = [
    SystemMessage(content="You are a well-trained and funny AI assistant uses Hinglish language.")
]

print("--------------Type 'exit' to quit.---------------------")
    
while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt.lower() == "exit":
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print("AI: " + response.content)
    print("----------------------------------------------")


print(messages)