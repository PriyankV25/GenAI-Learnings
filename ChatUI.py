import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Hinglish AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------
# Custom UI / Motion Design
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(120, 80, 255, 0.15),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(0, 200, 255, 0.12),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #080b14 0%,
                #0d1220 50%,
                #080b14 100%
            );

        color: #ffffff;
    }


    /* Hide Streamlit default elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* Main container */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }


    /* Animated title */
    .hero {
        text-align: center;
        padding: 25px 20px 20px 20px;
        animation: fadeDown 0.8s ease-out;
    }

    .hero-icon {
        font-size: 55px;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 8px 0 5px 0;

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

    .hero-subtitle {
        color: #a8b0c2;
        font-size: 16px;
        margin-bottom: 10px;
    }


    /* Status badge */
    .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;

        padding: 7px 14px;
        border-radius: 999px;

        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);

        color: #b8c0d4;
        font-size: 13px;

        animation: fadeIn 1s ease-out;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 12px #34d399;
        animation: pulse 1.8s infinite;
    }


    /* Chat message animation */
    [data-testid="stChatMessage"] {
        animation: messageIn 0.35s ease-out;
    }


    /* Chat input */
    [data-testid="stChatInput"] {
        animation: slideUp 0.5s ease-out;
    }


    /* Buttons */
    .stButton button {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.12);

        background: rgba(255,255,255,0.05);
        color: #ffffff;

        transition:
            transform 0.2s ease,
            background 0.2s ease,
            border-color 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        background: rgba(139,92,246,0.20);
        border-color: rgba(167,139,250,0.5);
    }


    /* Animations */

    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }

        50% {
            transform: translateY(-8px);
        }
    }

    @keyframes pulse {
        0% {
            transform: scale(1);
            opacity: 1;
        }

        50% {
            transform: scale(1.4);
            opacity: 0.5;
        }

        100% {
            transform: scale(1);
            opacity: 1;
        }
    }

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

    @keyframes messageIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
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
    <div class="hero">

        <div class="hero-icon">🤖</div>

        <div class="hero-title">
            Hinglish AI Assistant
        </div>

        <div class="hero-subtitle">
            A funny AI assistant powered by GPT-OSS 20B
        </div>

        <div class="status">
            <span class="status-dot"></span>
            AI Online
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Initialize Model
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    return init_chat_model(
        "openai/gpt-oss-20b",
        model_provider="groq",
        temperature=0.9
    )


model = load_model()


# ---------------------------------------------------------
# Initialize Conversation
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        SystemMessage(
            content=(
                "You are a well-trained and funny AI assistant "
                "uses Hinglish language."
            )
        )
    ]


# ---------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.markdown(message.content)


# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------

prompt = st.chat_input(
    "Type your message..."
)


# ---------------------------------------------------------
# Process User Message
# ---------------------------------------------------------

if prompt:

    # Same functionality as terminal version
    if prompt.lower() == "exit":

        st.session_state.messages.append(
            HumanMessage(content=prompt)
        )

        st.info("Chat ended.")

        st.stop()


    # Add user message
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)


    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("AI is thinking..."):

            response = model.invoke(
                st.session_state.messages
            )

            st.session_state.messages.append(
                AIMessage(content=response.content)
            )

        st.markdown(response.content)

