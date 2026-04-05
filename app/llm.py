import time

import openai

from app.config import (
    MAX_CONTEXT_CHARACTERS, OPENAI_MODELS, ANTHROPIC_MODELS, GROQ_MODELS,
    PROVIDER_BASE_URLS,
)

SYSTEM_PROMPT = (
    "You are an expert software engineer who has thoroughly read and understood "
    "the entire codebase provided below. Answer questions about the code accurately "
    "and concisely, referencing specific files and line content when relevant. "
    "If the answer is not found in the provided context, say so clearly rather than guessing."
)

_DEFAULTS = {
    "openai": OPENAI_MODELS[0],
    "anthropic": ANTHROPIC_MODELS[0],
    "groq": GROQ_MODELS[0],
}


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    used = 0

    for chunk in chunks:
        entry = f"### {chunk['source']}\n{chunk['text']}\n"
        if used + len(entry) > MAX_CONTEXT_CHARACTERS:
            break
        context_parts.append(entry)
        used += len(entry)

    context = "\n".join(context_parts)
    return f"CONTEXT FROM REPOSITORY:\n\n{context}\n\nQUESTION: {question}"


def _make_client(provider: str, api_key: str) -> openai.OpenAI:
    base_url = PROVIDER_BASE_URLS.get(provider)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def chat(
    question: str,
    chunks: list[dict],
    provider: str,
    api_key: str,
    model: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    if not api_key or not api_key.strip():
        return "Error: No API key provided."

    provider = provider.lower()
    if provider not in _DEFAULTS:
        return f"Error: Unknown provider '{provider}'."

    resolved_model = model or _DEFAULTS[provider]
    prompt = build_prompt(question, chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": prompt})

    client = _make_client(provider, api_key)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(model=resolved_model, messages=messages)
            return response.choices[0].message.content

        except openai.AuthenticationError as e:
            return f"Authentication error: {e}"

        except openai.RateLimitError as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 ** attempt * 2)

        except Exception as e:
            return f"Error calling {provider}: {e}"

    return f"Rate limit exceeded after retries: {last_error}"
