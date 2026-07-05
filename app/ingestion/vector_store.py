import os
import faiss
import numpy as np

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "/app/vector_store")

class VectorStore:
    def __init__(self, index_name="examgpt_index"):
        self.index_name = index_name
        self.index_path = os.path.join(VECTOR_STORE_DIR, f"{index_name}.index")
        self.index = None
        self.dimension = 384
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def save_index(self):
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        faiss.write_index(self.index, self.index_path)

    def add_embeddings(self, embeddings: list[list[float]]):
        if not embeddings:
            return
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        self.save_index()
        
    def search(self, query_embedding: list[float], top_k: int = 4):
        query_np = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_np, top_k)
        return distances, indices
