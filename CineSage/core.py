from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)


class MovieInfo(BaseModel):
    movie_name: Optional[str] = None
    release_year: Optional[str] = None
    genre: Optional[str] = None
    director: Optional[str] = None
    cast: List[str] = []
    main_characters: List[str] = []
    setting: Optional[str] = None
    main_premise: Optional[str] = None
    plot: Optional[str] = None
    key_events: List[str] = []
    main_conflict: Optional[str] = None
    resolution_outcome: Optional[str] = None
    important_locations: List[str] = []
    technology_science_concepts: List[str] = []
    key_themes: List[str] = []
    important_objects_elements: List[str] = []
    summary: Optional[str] = None


parser = PydanticOutputParser(pydantic_object=MovieInfo)



prompt = ChatPromptTemplate.from_messages([
   ( 
    "system",
"""
extract movie information from the provided paragraph and return it in a structured format.
{format_instructions}
"""
    ),
    (
        "human",
        """
{paragraph}
"""
    )
])

para = input("Enter a paragraph : ")

final_prompt = prompt.invoke({"paragraph": para, 
                              "format_instructions": parser.get_format_instructions()})
response = model.invoke(final_prompt)
print(response.content)