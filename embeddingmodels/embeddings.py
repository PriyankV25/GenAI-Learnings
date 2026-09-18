from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import sys

load_dotenv()

# sys.stdout.reconfigure(encoding="utf-8")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=64
)

vector = embeddings.embed_query("you are going to learn Gen AI")

print(vector)