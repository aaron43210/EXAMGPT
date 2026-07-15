"""
LLM and embedding model setup via Hugging Face Inference API.
Uses langchain_huggingface for cloud-based model access.
"""
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings, ChatHuggingFace
from app.core.config import get_settings


def get_llm():
    settings = get_settings()

    # Use Mistral-7B — fast, reliable, and rarely rate-limited on the free HuggingFace router
    # Qwen2.5-72B hits 429 Too Many Requests frequently due to high demand
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        temperature=0.2,
        max_new_tokens=4096,
        task="conversational",
    )
    return ChatHuggingFace(llm=endpoint)


def get_embeddings():
    settings = get_settings()
    return HuggingFaceEndpointEmbeddings(
        model=settings.EMBEDDING_MODEL,
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
    )
