# from dotenv import load_dotenv

# load_dotenv()

# import os
# from langchain.chat_models import init_chat_model
# model = init_chat_model("gpt-4.1")
# print(model)        
# response = model.invoke("what is cricket?")
# print(response)

from dotenv import load_dotenv

load_dotenv()


import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)

response = model.invoke("What is cricket?")

print(response.content)