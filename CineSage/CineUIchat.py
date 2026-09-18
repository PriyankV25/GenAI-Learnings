
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Information Extractor",
    page_icon="🔎",
    layout="centered"
)


# ---------------------------------------------------------
# Custom UI
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Background */
    .stApp {
        background:
            radial-gradient(
                circle at 15% 15%,
                rgba(99, 102, 241, 0.16),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(6, 182, 212, 0.12),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #080b14 0%,
                #0d1220 50%,
                #080b14 100%
            );
    }

    /* Main container */
    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .header {
        text-align: center;
        animation: fadeDown 0.8s ease-out;
        margin-bottom: 2rem;
    }

    .icon {
        font-size: 52px;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }

    .title {
        font-size: 42px;
        font-weight: 800;
        margin-top: 8px;

        background: linear-gradient(
            90deg,
            #ffffff,
            #a78bfa,
            #67e8f9,
            #ffffff
        );

        background-size: 300% auto;

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        animation: gradientMove 5s linear infinite;
    }

    .subtitle {
        color: #a8b0c2;
        font-size: 16px;
        margin-top: 8px;
    }

    /* Input label */
    .input-label {
        color: #d8deeb;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Text area */
    textarea {
        background: rgba(255, 255, 255, 0.045) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        color: white !important;
        transition: all 0.25s ease;
    }

    textarea:focus {
        border-color: rgba(167, 139, 250, 0.7) !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.12) !important;
    }

    /* Extract button */
    .stButton button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(167, 139, 250, 0.35);

        background: linear-gradient(
            135deg,
            rgba(124, 58, 237, 0.8),
            rgba(79, 70, 229, 0.8)
        );

        color: white;
        font-weight: 600;

        padding: 0.65rem;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25);
    }

    /* Result container */
    .result-header {
        margin-top: 2rem;
        margin-bottom: 1rem;

        color: #e5e7eb;
        font-size: 20px;
        font-weight: 700;

        animation: fadeIn 0.5s ease-out;
    }

    /* Animations */

    @keyframes fadeDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
        }

        to {
            opacity: 1;
        }
    }

    @keyframes float {
        0%, 100% {
            transform: translateY(0);
        }

        50% {
            transform: translateY(-7px);
        }
    }

    @keyframes gradientMove {
        0% {
            background-position: 0% center;
        }

        50% {
            background-position: 100% center;
        }

        100% {
            background-position: 0% center;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    """
    <div class="header">

        <div class="icon">🔎</div>

        <div class="title">
            Information Extractor
        </div>

        <div class="subtitle">
            Extract useful information and generate a quick summary
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    return init_chat_model(
        "openai/gpt-oss-20b",
        model_provider="groq"
    )


model = load_model()


# ---------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert information extraction assistant.

Your task is to analyze the provided paragraph and extract useful,
factual information from it.

Follow these rules strictly:

1. Extract information ONLY from the provided paragraph.
2. Do NOT invent, assume, or add information that is not present.
3. If information is not mentioned, write "Not mentioned".
4. Keep the extracted information concise and clear.
5. Separate characters from actors/cast. Only identify an actor as
   cast if the paragraph explicitly mentions the actor.
6. Identify important events, locations, concepts, themes, and
   other useful information when they are present.
7. Create a quick summary of the paragraph in 2-3 sentences.
8. Return the answer using exactly the following structure.

MOVIE INFORMATION
- Movie Name:
- Release Year:
- Genre:
- Director:
- Cast:
- Main Characters:
- Setting:

STORY INFORMATION
- Main Premise:
- Plot:
- Key Events:
- Main Conflict:
- Resolution / Outcome:

ADDITIONAL INFORMATION
- Important Locations:
- Technology / Science Concepts:
- Key Themes:
- Important Objects / Elements:

QUICK SUMMARY
- Summary:

Do not provide explanations about how you extracted the information.
Only provide the structured result.
"""
    ),
    (
        "human",
        """
Extract useful information from the following paragraph:

{paragraph}
"""
    )
])


# ---------------------------------------------------------
# Paragraph Input
# ---------------------------------------------------------

st.markdown(
    '<div class="input-label">Enter your paragraph</div>',
    unsafe_allow_html=True
)

para = st.text_area(
    label="paragraph",
    label_visibility="collapsed",
    placeholder="Paste your paragraph here...",
    height=220
)


# ---------------------------------------------------------
# Extract Button
# ---------------------------------------------------------

extract = st.button(
    "🔍 Extract Information"
)


# ---------------------------------------------------------
# Generate Result
# ---------------------------------------------------------

if extract:

    if not para.strip():

        st.warning("Please enter a paragraph.")

    else:

        with st.spinner("Analyzing paragraph..."):

            final_prompt = prompt.invoke({
                "paragraph": para
            })

            response = model.invoke(final_prompt)

        st.markdown(
            '<div class="result-header">Extracted Information</div>',
            unsafe_allow_html=True
        )

        st.markdown(response.content)

