import hashlib
from typing import Callable

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

_embedding_model: SentenceTransformer | None = None


def _get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None or _embedding_model.get_sentence_embedding_dimension() is None:
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def init_store(persist_dir: str = CHROMA_PERSIST_DIR) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=persist_dir)


def get_collection_name(repo_path: str) -> str:
    digest = hashlib.md5(repo_path.encode()).hexdigest()[:12]
    return f"repo_{digest}"


def collection_exists(client: chromadb.ClientAPI, name: str) -> bool:
    return any(c.name == name for c in client.list_collections())


def embed_and_store(
    client: chromadb.ClientAPI,
    collection_name: str,
    chunks: list[dict],
    embedding_model: str = EMBEDDING_MODEL,
    progress_callback: Callable[[int, int], None] | None = None,
    overwrite: bool = False,
) -> chromadb.Collection:
    """Embed all chunks and store in ChromaDB. Returns the collection."""
    if overwrite and collection_exists(client, collection_name):
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = _get_embedding_model(embedding_model)
    total = len(chunks)

    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = chunks[batch_start: batch_start + EMBEDDING_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        ids = [f"{c['source']}::chunk_{c['chunk_index']}" for c in batch]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in batch]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        if progress_callback:
            progress_callback(min(batch_start + EMBEDDING_BATCH_SIZE, total), total)

    return collection


def retrieve(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 8,
) -> list[dict]:
    """Embed the query and return top-N matching chunks."""
    model = _get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "score": 1.0 - dist,  # cosine distance → similarity
        })

    return chunks
