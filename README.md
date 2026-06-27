# 🤖 HuggingFace Chatbot

A conversational AI chatbot built with **Streamlit** and **LangChain**, powered by open-source models from HuggingFace. No GPU required — it calls the HuggingFace Inference API remotely.

<img width="1603" height="854" alt="Screenshot 2026-06-27 at 9 19 07 PM" src="https://github.com/user-attachments/assets/db033417-b48e-4f23-8322-b1c4b1c79bd2" />

<img width="1598" height="835" alt="Screenshot 2026-06-27 at 9 19 27 PM" src="https://github.com/user-attachments/assets/6df15c49-7cf3-46dc-87f7-9f8aabc52b3e" />


## Features

- **3 open-source models** — Llama 3 8B, Mistral 7B, Zephyr 7B
- **Live streaming** responses (token-by-token, like ChatGPT)
- **Adjustable settings** — temperature, max tokens, system prompt
- **Full conversation memory** across the session
- **Clear chat** to reset and start fresh

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM orchestration | LangChain (`langchain-huggingface`) |
| Models | HuggingFace Inference API |
| Language | Python 3.10+ |

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/vishalinis/Ai-HuggingFace-Chatbot.git
cd Ai-HuggingFace-Chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your HuggingFace API token

Get a free token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Create a `.env` file in the project root:

```
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set entry point to `app.py`
4. Under **Advanced settings → Secrets**, add:

```toml
HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"
```

5. Click **Deploy** — live in ~2 minutes

---

## Project Structure

```
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
├── .gitignore        # Excludes .env and cache files
└── README.md
```

## Models Used

| Model | HuggingFace Repo | Notes |
|---|---|---|
| Llama 3 8B Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | Requires HF access request |
| Mistral 7B Instruct | `mistralai/Mistral-7B-Instruct-v0.3` | Open access |
| Zephyr 7B Beta | `HuggingFaceH4/zephyr-7b-beta` | Open access |

> **Note:** Llama 3 requires you to accept Meta's license on the HuggingFace model page before your token will work with it.
