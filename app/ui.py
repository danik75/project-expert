import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from app.config import OPENAI_MODELS, ANTHROPIC_MODELS, GROQ_MODELS, N_RETRIEVE_RESULTS, PROVIDER_ENV_KEYS
from app.ingestor import ingest_repo
from app.vectorstore import init_store, get_collection_name, collection_exists, embed_and_store, retrieve
from app.llm import chat

_chroma_client = None

PROVIDER_MODELS = {
    "OpenAI": OPENAI_MODELS,
    "Anthropic": ANTHROPIC_MODELS,
    "Groq": GROQ_MODELS,
}


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = init_store()
    return _chroma_client


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Project Expert") as app:
        gr.Markdown("# Project Expert\nPoint at a local project folder and chat with it using RAG + LLM.")

        collection_state = gr.State(None)
        history_state = gr.State([])

        # ── Section 1: Setup ────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("## 1. Project & Model Setup")

            repo_input = gr.Textbox(
                label="Project Path",
                placeholder="e.g. /Users/me/projects/my-app  or  ~/projects/my-app",
            )

            with gr.Row():
                provider_radio = gr.Radio(
                    choices=list(PROVIDER_MODELS.keys()),
                    value="OpenAI",
                    label="LLM Provider",
                )
                model_dropdown = gr.Dropdown(
                    choices=OPENAI_MODELS,
                    value=OPENAI_MODELS[0],
                    label="Model",
                )

            force_reingest = gr.Checkbox(label="Force re-ingest (ignore cached embeddings)", value=False)

            load_btn = gr.Button("Load Project", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, lines=3)

        # ── Section 2: Chat ─────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("## 2. Chat")

            chatbot = gr.Chatbot(label="Conversation", height=450)

            with gr.Row():
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="Ask anything about the project...",
                    scale=4,
                    interactive=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, interactive=False)

            clear_btn = gr.Button("Clear Chat", interactive=False)

        # ── Event: provider change → update model dropdown ───────────────────
        def on_provider_change(provider: str):
            models = PROVIDER_MODELS.get(provider, OPENAI_MODELS)
            return gr.update(choices=models, value=models[0])

        provider_radio.change(on_provider_change, inputs=provider_radio, outputs=model_dropdown)

        def _resolve_api_key(provider: str) -> str:
            env_var = PROVIDER_ENV_KEYS.get(provider.lower(), "")
            return os.environ.get(env_var, "")

        # ── Event: Load Project ──────────────────────────────────────────────
        def handle_load_repo(source: str, provider: str, model: str, force: bool):
            disabled = (gr.update(interactive=False),) * 3

            if not source.strip():
                yield gr.update(), "Please enter a project path.", None, *disabled
                return

            client = _get_client()

            yield gr.update(), "Scanning project files...", None, *disabled

            try:
                chunks, repo_path = ingest_repo(source)
            except ValueError as e:
                yield gr.update(), f"Error: {e}", None, *disabled
                return

            collection_name = get_collection_name(repo_path)
            file_count = len(set(c["source"] for c in chunks))

            if collection_exists(client, collection_name) and not force:
                collection = client.get_collection(collection_name)
                status = (
                    f"Loaded cached embeddings.\n"
                    f"Path: {repo_path}\n"
                    f"Files: {file_count} | Chunks: {len(chunks)} | Collection: {collection_name}"
                )
                yield gr.update(), status, collection, *(gr.update(interactive=True),) * 3
                return

            yield (
                gr.update(),
                f"Found {file_count} files, {len(chunks)} chunks. Embedding...",
                None,
                *disabled,
            )

            try:
                collection = embed_and_store(client, collection_name, chunks, overwrite=force)
            except Exception as e:
                yield gr.update(), f"Embedding error: {e}", None, *disabled
                return

            status = (
                f"Ready!\n"
                f"Path: {repo_path}\n"
                f"Files: {file_count} | Chunks: {len(chunks)} | Collection: {collection_name}"
            )
            yield gr.update(), status, collection, *(gr.update(interactive=True),) * 3

        load_btn.click(
            handle_load_repo,
            inputs=[repo_input, provider_radio, model_dropdown, force_reingest],
            outputs=[chatbot, status_box, collection_state, question_input, send_btn, clear_btn],
        )

        # ── Event: Send message ──────────────────────────────────────────────
        def handle_chat(user_msg: str, history: list, collection, provider: str, model: str):
            if not user_msg.strip():
                return history, history, ""
            if collection is None:
                history = history + [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": "Please load a project first."},
                ]
                return history, history, ""

            # Convert messages-format history to (user, assistant) pairs for llm.chat()
            pairs = []
            msgs = [m for m in history if m["role"] in ("user", "assistant")]
            for i in range(0, len(msgs) - 1, 2):
                if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                    pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))

            chunks = retrieve(collection, user_msg, n_results=N_RETRIEVE_RESULTS)
            response = chat(
                question=user_msg,
                chunks=chunks,
                provider=provider.lower(),
                api_key=_resolve_api_key(provider),
                model=model,
                history=pairs,
            )

            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ]
            return history, history, ""

        send_btn.click(
            handle_chat,
            inputs=[question_input, history_state, collection_state, provider_radio, model_dropdown],
            outputs=[chatbot, history_state, question_input],
        )
        question_input.submit(
            handle_chat,
            inputs=[question_input, history_state, collection_state, provider_radio, model_dropdown],
            outputs=[chatbot, history_state, question_input],
        )

        # ── Event: Clear chat ────────────────────────────────────────────────
        clear_btn.click(lambda: ([], []), outputs=[chatbot, history_state])

    return app
