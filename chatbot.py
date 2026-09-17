
from dotenv import load_dotenv

load_dotenv()

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq",
    temperature=0.9
)

messages = []

print("--------------Type 'exit' to quit.---------------------")
    
while True:
    prompt = input("You: ")
    messages.append(prompt)
    if prompt.lower() == "exit":
        break
    response = model.invoke(messages)
    messages.append(response.content)

    print("AI: " + response.content)
    print("----------------------------------------------")