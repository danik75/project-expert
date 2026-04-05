# Project Expert

A local RAG (Retrieval-Augmented Generation) chatbot that turns any codebase into an interactive expert. Point it at a project folder, and it will index the source files and let you chat with an LLM that has full context of your code.

## How it works

1. You provide a local path to any project directory
2. The app scans and chunks all source files into a local vector database (ChromaDB)
3. When you ask a question, the most relevant chunks are retrieved and injected into the LLM prompt
4. The LLM answers with full awareness of the codebase

Embeddings are cached locally, so reloading the same project is instant.

---

## Supported LLM Providers

| Provider  | Models available |
|-----------|-----------------|
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-4-turbo |
| **Anthropic** | claude-opus-4-6, claude-sonnet-4-5, claude-haiku-4-5 |
| **Groq** | Qwen 3 32B, Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B, Mixtral 8x7B, Llama 4 Scout |

All three providers use the OpenAI-compatible chat completions API, so no extra SDKs are needed beyond `openai`.

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager

Install `uv` if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/danik75/project-expert.git
cd project-expert
```

### 2. Install dependencies

```bash
uv sync
```

This creates a virtual environment and installs all dependencies automatically.

### 3. Configure API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and add the keys for the providers you want to use:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

You only need to fill in the key for the provider(s) you plan to use. **Groq is free** and a great starting point.

---

## Getting a Groq API key (free)

Groq offers a generous free tier with access to fast open-source models like Qwen, Llama, and Mixtral — no credit card required.

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with your Google or GitHub account
3. Navigate to **API Keys** in the left sidebar
4. Click **Create API Key**, give it a name, and copy it
5. Paste it as `GROQ_API_KEY` in your `.env` file

---

## Getting an OpenAI API key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click **Create new secret key**
4. Copy the key and paste it as `OPENAI_API_KEY` in your `.env` file

---

## Getting an Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign in or create an account
3. Navigate to **Settings → API Keys**
4. Click **Create Key**, copy it, and paste it as `ANTHROPIC_API_KEY` in your `.env` file

---

## Running the app

```bash
uv run main.py
```

Then open your browser at **http://127.0.0.1:7860**

---

## Usage

1. **Project Path** — enter the absolute path to any local project folder, e.g. `/Users/me/projects/my-app`
2. **LLM Provider** — choose OpenAI, Anthropic, or Groq
3. **Model** — select a model from the dropdown (updates automatically per provider)
4. Click **Load Project** — the app scans and indexes your files (first time takes a moment; subsequent loads use the cache)
5. Once the status shows **Ready**, start chatting in the chat panel below

### Tips

- Use **Force re-ingest** if you've made changes to the project and want the index refreshed
- Groq's **Qwen 3 32B** and **Llama 3.3 70B** are excellent free models for code questions
- The app indexes: `.py` `.js` `.ts` `.go` `.rs` `.rb` `.java` `.c` `.cpp` `.md` `.json` `.yaml` `.toml` `.sql` `.sh` and more
- Binary files, `node_modules/`, `.venv/`, and `dist/` folders are automatically skipped

---

## Project structure

```
project-expert/
├── main.py              # Entry point
├── pyproject.toml       # Dependencies
├── .env                 # Your API keys (never committed)
└── app/
    ├── config.py        # Constants and model lists
    ├── ingestor.py      # File discovery and chunking
    ├── vectorstore.py   # ChromaDB embedding and retrieval
    ├── llm.py           # LLM call dispatcher
    └── ui.py            # Gradio interface
```
