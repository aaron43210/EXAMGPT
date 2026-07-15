"""
LLM and embedding model setup via Hugging Face Inference API.
Uses langchain_huggingface for cloud-based model access.
"""
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings, ChatHuggingFace
from app.core.config import get_settings


def get_llm():
    settings = get_settings()
    
    # Force Llama 3 to override any stale Hugging Face Secrets
    model_id = "Qwen/Qwen2.5-72B-Instruct"

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
