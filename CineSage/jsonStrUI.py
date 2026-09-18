
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser


# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)


# --------------------------------------------------
# Pydantic Model
# --------------------------------------------------

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


# --------------------------------------------------
# Output Parser
# --------------------------------------------------

parser = PydanticOutputParser(
    pydantic_object=MovieInfo
)


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
extract movie information from the provided paragraph
and return it in a structured format.

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


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🎬 Movie Information Extractor")

st.write(
    "Enter a paragraph and extract structured movie information."
)


# --------------------------------------------------
# Paragraph Input
# --------------------------------------------------

para = st.text_area(
    "Enter a paragraph:",
    height=200,
    placeholder="Paste your movie paragraph here..."
)


# --------------------------------------------------
# Extract Button
# --------------------------------------------------

if st.button("Extract Information"):

    if not para.strip():

        st.warning("Please enter a paragraph.")

    else:

        with st.spinner("Extracting information..."):

            final_prompt = prompt.invoke({
                "paragraph": para,
                "format_instructions":
                    parser.get_format_instructions()
            })

            response = model.invoke(final_prompt)

            movie_info = parser.parse(response.content)


        # --------------------------------------------------
        # Display Result
        # --------------------------------------------------

        st.subheader("Extracted Movie Information")

        st.write("**Movie Name:**", movie_info.movie_name)

        st.write("**Release Year:**", movie_info.release_year)

        st.write("**Genre:**", movie_info.genre)

        st.write("**Director:**", movie_info.director)

        st.write("**Cast:**")
        for person in movie_info.cast:
            st.write(f"- {person}")

        st.write("**Main Characters:**")
        for character in movie_info.main_characters:
            st.write(f"- {character}")

        st.write("**Setting:**", movie_info.setting)

        st.write("**Main Premise:**", movie_info.main_premise)

        st.write("**Plot:**", movie_info.plot)

        st.write("**Key Events:**")
        for event in movie_info.key_events:
            st.write(f"- {event}")

        st.write("**Main Conflict:**", movie_info.main_conflict)

        st.write(
            "**Resolution / Outcome:**",
            movie_info.resolution_outcome
        )

        st.write("**Important Locations:**")
        for location in movie_info.important_locations:
            st.write(f"- {location}")

        st.write("**Technology / Science Concepts:**")
        for concept in movie_info.technology_science_concepts:
            st.write(f"- {concept}")

        st.write("**Key Themes:**")
        for theme in movie_info.key_themes:
            st.write(f"- {theme}")

        st.write("**Important Objects / Elements:**")
        for element in movie_info.important_objects_elements:
            st.write(f"- {element}")

        st.write("**Quick Summary:**")
        st.write(movie_info.summary)

