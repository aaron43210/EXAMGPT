"""
LLM and embedding model setup via Ollama.
Uses langchain_ollama (modern, non-deprecated API).
"""
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from app.core.config import get_settings


def get_llm():
    settings = get_settings()
    return HuggingFaceEndpoint(
        repo_id=settings.LLM_MODEL,
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        temperature=0.2,
        task="text-generation",
    )


def get_embeddings():
    settings = get_settings()
    return HuggingFaceEndpointEmbeddings(
        model=settings.EMBEDDING_MODEL,
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
    )
