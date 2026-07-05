from app.agents.llm_setup import get_embeddings


class Embedder:
    def __init__(self):
        self.model = get_embeddings()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)
