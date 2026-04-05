SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".rb", ".java", ".c", ".cpp", ".h", ".cs",
    ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".sh",
    ".env.example", ".dockerfile", ".tf", ".sql", ".html", ".css",
    ".vue", ".svelte", ".kt", ".swift", ".scala", ".r", ".m",
})

SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".idea", ".vscode", "vendor", "target",
    ".mypy_cache", ".pytest_cache", "eggs", ".eggs",
})

MAX_FILE_SIZE_BYTES: int = 500_000       # 500 KB
MAX_CHUNK_CHARACTERS: int = 1500
CHUNK_OVERLAP_CHARACTERS: int = 150

CHROMA_PERSIST_DIR: str = ".chroma_db"
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
EMBEDDING_BATCH_SIZE: int = 64

OPENAI_MODELS: list[str] = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
ANTHROPIC_MODELS: list[str] = ["claude-opus-4-6", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"]
GROQ_MODELS: list[str] = [
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# OpenAI-compatible base URLs (None = default OpenAI endpoint)
PROVIDER_BASE_URLS: dict[str, str | None] = {
    "openai": None,
    "anthropic": "https://api.anthropic.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}

# Corresponding .env variable names
PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
}

MAX_CONTEXT_CHARACTERS: int = 12_000
N_RETRIEVE_RESULTS: int = 8
