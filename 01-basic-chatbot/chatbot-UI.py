import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

# Page Configuration
st.set_page_config(page_title="Persona AI Chatbot", page_icon="🤖", layout="centered")

# Custom Dark Theme Styling
st.markdown("""
<style>
    /* Main Background & Text Color */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Header Styling */
    h1 {
        color: #00ADB5 !important;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin-bottom: 20px;
    }

    /* Radio / Mode Selection Styling */
    div[data-testid="stRadio"] > label {
        color: #00ADB5 !important;
        font-weight: 600;
    }

    /* Chat Messages Container */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }

    /* User Chat Bubble */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #1E232A;
        border: 1px solid #2D3748;
    }

    /* Assistant Chat Bubble */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #161B22;
        border: 1px solid #00ADB5;
    }

    /* Input Box Styling */
    .stChatInputContainer textarea {
        background-color: #1A1D24 !important;
        color: #FFFFFF !important;
        border: 1px solid #00ADB5 !important;
        border-radius: 10px !important;
    }
    
    /* Hide Streamlit Default Header/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

load_dotenv()

@st.cache_resource
def get_model():
    return ChatMistralAI(
        model="open-mistral-7b",
        temperature=0.7
    )

model = get_model()

st.title("🤖 Persona AI Chatbot")

# Mode Selection
mode_options = {
    "Happy Agent": "You are a Happy chat Agent. Response user in happy mode",
    "Sad Agent": "You are a Sad chat Agent. Response user in Sad mode",
    "Angry Agent": "You are a Angry chat Agent. Response user in angry mode"
}

selected_mode_label = st.radio(
    "Select Agent Personality:",
    options=list(mode_options.keys()),
    horizontal=True
)

mode = mode_options[selected_mode_label]

# Session State Initialization
if "messages" not in st.session_state or st.session_state.get("current_mode") != selected_mode_label:
    st.session_state.current_mode = selected_mode_label
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

# Render existing chat history (Skipping SystemMessage)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg.content)

# User Input Handling
if user_input := st.chat_input("Type your message here..."):
    # Render User Message
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)
    
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Get and Render AI Response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
            st.write(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))