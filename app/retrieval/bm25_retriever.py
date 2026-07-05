from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []

    def build_index(self, texts: list[str], documents: list):
        if not texts:
            return
        self.documents = documents
        tokenized_corpus = [doc.split(" ") for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 4):
        if not self.bm25:
            return [], []
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices, [scores[i] for i in top_indices]
