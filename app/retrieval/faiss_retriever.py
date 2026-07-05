from app.ingestion.vector_store import VectorStore
from app.ingestion.embedder import Embedder

class FAISSRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = Embedder()

    def search(self, query: str, top_k: int = 4):
        query_embedding = self.embedder.embed_query(query)
        distances, indices = self.vector_store.search(query_embedding, top_k)
        return indices[0], distances[0]
