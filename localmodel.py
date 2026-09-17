import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import sys

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={
        "token": os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    },
    pipeline_kwargs={
        "max_new_tokens": 512,
        "do_sample": False,
        "repetition_penalty": 1.03,
    },
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is data science?")

print(response.content)