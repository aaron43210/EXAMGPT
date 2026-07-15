"""
LLM and embedding model setup via Hugging Face Inference API.
Uses langchain_huggingface for cloud-based model access.
"""
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings, ChatHuggingFace
from app.core.config import get_settings


def get_llm():
    settings = get_settings()
    
    # Llama-3.1-8B-Instruct — verified chat model on HuggingFace free-tier router
    # Mistral-7B-v0.3 is not registered as a chat model on the HF router
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

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
