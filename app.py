"""
HuggingFace Chatbot
-------------------
A conversational AI chatbot built with Streamlit and LangChain.
Connects to the HuggingFace Inference API — no local GPU required.
Supports multiple open-source models with live streaming responses.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ── Environment & Secrets ─────────────────────────────────────────────────────
# Locally:          create a .env file with HUGGINGFACEHUB_API_TOKEN=hf_xxx
# Streamlit Cloud:  add the token under App Settings → Secrets
load_dotenv()
try:
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except Exception:
    pass  # no secrets.toml present locally — falls back to .env

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HuggingFace Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Supported Models ──────────────────────────────────────────────────────────
# Maps display names to HuggingFace repo IDs used by the Inference API.
# Add or swap models here — they must support the text-generation task.
MODELS = {
    "Llama 3 8B Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "Zephyr 7B Beta": "HuggingFaceH4/zephyr-7b-beta",
}


# ── Model Loader ──────────────────────────────────────────────────────────────
# @st.cache_resource caches the model per (repo_id, temperature, max_new_tokens)
# so it is only re-initialized when the user changes a setting, not on every rerun.
@st.cache_resource
def load_model(repo_id: str, temperature: float, max_new_tokens: int):
    """Connect to a HuggingFace Inference API endpoint and wrap it for chat."""
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,       # required when temperature > 0
    )
    # ChatHuggingFace wraps the endpoint to support the chat message format
    # (SystemMessage / HumanMessage / AIMessage) used by LangChain
    return ChatHuggingFace(llm=endpoint)


# ── Sidebar — Settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Model selector — switching reloads the cached model automatically
    model_name = st.selectbox("Model", list(MODELS.keys()))

    # Temperature controls randomness: 0.1 = deterministic, 1.0 = very creative
    temperature = st.slider(
        "Temperature",
        min_value=0.1, max_value=1.0, value=0.7, step=0.1,
        help="Higher = more creative · Lower = more focused",
    )

    # Limits how many tokens the model generates per response
    max_tokens = st.slider(
        "Max Tokens",
        min_value=128, max_value=1024, value=512, step=128,
    )

    # System prompt sets the assistant's persona for the whole conversation
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        height=100,
        help="Defines the assistant's personality and behavior.",
    )

    st.divider()

    # Clears the conversation and starts fresh with the current system prompt
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = [SystemMessage(content=system_prompt)]
        st.rerun()

    st.divider()

    # Live stats shown at the bottom of the sidebar
    msg_count = len([
        m for m in st.session_state.get("chat_history", [])
        if isinstance(m, (HumanMessage, AIMessage))
    ])
    st.caption(f"Model: `{MODELS[model_name].split('/')[-1]}`")
    st.caption(f"Messages this session: **{msg_count}**")


# ── Model Init ────────────────────────────────────────────────────────────────
model = load_model(MODELS[model_name], temperature, max_tokens)

# ── Session State ─────────────────────────────────────────────────────────────
# chat_history holds the full conversation as LangChain message objects.
# It always starts with a SystemMessage so the model knows its role.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content=system_prompt)]

# Sync system prompt live if the user edits it in the sidebar mid-conversation
if isinstance(st.session_state.chat_history[0], SystemMessage):
    st.session_state.chat_history[0] = SystemMessage(content=system_prompt)

# ── Page Header ───────────────────────────────────────────────────────────────
st.title("🤖 HuggingFace Chatbot")
st.caption("Powered by LangChain · Meta Llama · Mistral · Zephyr")

# ── Empty State ───────────────────────────────────────────────────────────────
# Shown only when no messages exist yet — gives new users a clear starting cue
visible = [m for m in st.session_state.chat_history if isinstance(m, (HumanMessage, AIMessage))]
if not visible:
    st.markdown(
        """
        <div style="text-align:center; color:gray; padding:3rem 0;">
            <h3>Start a conversation</h3>
            <p>Ask me anything — I'm ready to help.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Chat History Display ──────────────────────────────────────────────────────
# Replays the full conversation on every Streamlit rerun.
# SystemMessage is intentionally skipped — it's internal context, not UI.
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)   # markdown renders code blocks, bold, etc.

# ── Chat Input & Response ─────────────────────────────────────────────────────
user_input = st.chat_input("Type your message…")

if user_input:
    # Append the user's message to history and display it immediately
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        # Stream the response token-by-token using model.stream()
        # st.write_stream() renders each chunk as it arrives (like ChatGPT)
        with st.chat_message("assistant"):
            response = st.write_stream(model.stream(st.session_state.chat_history))
        # Save the completed response to history for the next turn
        st.session_state.chat_history.append(AIMessage(content=response))
    except Exception as e:
        st.error(f"Error getting response: {e}")
