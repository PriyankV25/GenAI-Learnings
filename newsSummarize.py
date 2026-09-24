from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.stdout.reconfigure(encoding="utf-8")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

search_tool = TavilySearch(max_results=5)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Summarize the following latest news into clear,
concise bullet points.

Focus on:
- What happened
- Important organizations or people
- Important technical developments
- Why the news matters
- Relevant dates

News: {news} 
    
    """
)

chain = prompt | model | StrOutputParser()

news_result = search_tool.run("Latest AI news of september 2026")

result = chain.invoke({"news": news_result})

print("\n==============================")
print("        AI NEWS SUMMARY")
print("==============================\n")

print(result)