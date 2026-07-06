"""
LLM and embedding model setup via Hugging Face Inference API.
Uses langchain_huggingface for cloud-based model access.
"""
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings, ChatHuggingFace
from app.core.config import get_settings


def get_llm():
    settings = get_settings()
    endpoint = HuggingFaceEndpoint(
        repo_id=settings.LLM_MODEL,
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        temperature=0.2,
        task="conversational",
    )
    return ChatHuggingFace(llm=endpoint)


def get_embeddings():
    settings = get_settings()
    return HuggingFaceEndpointEmbeddings(
        model=settings.EMBEDDING_MODEL,
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
    )
