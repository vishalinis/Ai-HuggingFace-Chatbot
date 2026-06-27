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
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
    # (HumanMessage / AIMessage) used by LangChain
    return ChatHuggingFace(llm=endpoint)


# ── Prompt Template ───────────────────────────────────────────────────────────
# Structure of every call to the model:
#   1. system   — persona/behavior, injected fresh from the sidebar each time
#   2. history  — all past HumanMessage + AIMessage pairs (MessagesPlaceholder)
#   3. human    — the current user message
#
# MessagesPlaceholder is the slot where the full chat history is inserted at
# runtime. This keeps the template structure separate from the conversation data.
prompt_template = ChatPromptTemplate([
    ("system", "{system_prompt}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{query}"),
])


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

    # System prompt is now a template variable — changing it takes effect
    # immediately on the next message without touching chat_history
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        height=100,
        help="Defines the assistant's personality and behavior.",
    )

    st.divider()

    # chat_history no longer holds a SystemMessage, so clearing is just []
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    msg_count = len(st.session_state.get("chat_history", []))
    st.caption(f"Model: `{MODELS[model_name].split('/')[-1]}`")
    st.caption(f"Messages this session: **{msg_count}**")


# ── Model + Chain ─────────────────────────────────────────────────────────────
# LCEL chain: template fills the prompt → model generates the response
model = load_model(MODELS[model_name], temperature, max_tokens)
chain = prompt_template | model

# ── Session State ─────────────────────────────────────────────────────────────
# chat_history stores only HumanMessage + AIMessage pairs.
# The SystemMessage is no longer stored here — it lives in the template
# and is injected fresh from the sidebar variable on every call.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Page Header ───────────────────────────────────────────────────────────────
st.title("🤖 HuggingFace Chatbot")
st.caption("Powered by LangChain · Meta Llama · Mistral · Zephyr")

# ── Empty State ───────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
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
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# ── Chat Input & Response ─────────────────────────────────────────────────────
user_input = st.chat_input("Type your message…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        # chain.stream() fills the template with:
        #   system_prompt → from the sidebar (always fresh)
        #   chat_history  → past Human+AI pairs from session state
        #   query         → the current user message
        with st.chat_message("assistant"):
            response = st.write_stream(
                chain.stream({
                    "system_prompt": system_prompt,
                    "chat_history": st.session_state.chat_history,
                    "query": user_input,
                })
            )

        # Save both sides of this turn to history for the next call
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

    except Exception as e:
        st.error(f"Error getting response: {e}")
