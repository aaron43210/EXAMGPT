from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.faiss_retriever import FAISSRetriever
from langchain_core.documents import Document


class HybridRetriever:
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.faiss_retriever = FAISSRetriever()
        self.documents = []

    def add_documents(self, docs: list[Document]):
        if not docs:
            return
        self.documents.extend(docs)
        texts = [doc.page_content for doc in docs]

        # Rebuild BM25
        self.bm25_retriever.build_index(
            [d.page_content for d in self.documents], self.documents
        )

        # Build FAISS
        embeddings = self.faiss_retriever.embedder.embed_documents(texts)
        self.faiss_retriever.vector_store.add_embeddings(embeddings)

    def retrieve(self, query: str, top_k: int = 4, doc_type_filter: str = None) -> list[Document]:
        """Retrieve documents with optional doc_type filtering."""
        if not self.documents:
            return []

        # Filter documents by type if specified
        if doc_type_filter:
            filtered_docs = [
                d for d in self.documents
                if d.metadata.get("doc_type", "notes") == doc_type_filter
            ]
            if not filtered_docs:
                return []
            # Search within filtered docs using BM25
            bm25_ret = BM25Retriever()
            bm25_ret.build_index(
                [d.page_content for d in filtered_docs], filtered_docs
            )
            bm25_indices, _ = bm25_ret.search(query, min(top_k, len(filtered_docs)))
            results = [filtered_docs[i] for i in bm25_indices if i < len(filtered_docs)]
            return results[:top_k]

        # Default: full hybrid retrieval (BM25 + FAISS with RRF)
        bm25_indices, _ = self.bm25_retriever.search(query, top_k)
        faiss_indices, _ = self.faiss_retriever.search(query, top_k)

        fused_scores = {}
        for rank, doc_idx in enumerate(bm25_indices):
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + 1 / (rank + 60)

        for rank, doc_idx in enumerate(faiss_indices):
            if doc_idx != -1 and doc_idx < len(self.documents):
                fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + 1 / (rank + 60)

        sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        top_indices = sorted_indices[:top_k]

        return [self.documents[idx] for idx in top_indices if idx < len(self.documents)]


# Global instance
hybrid_retriever = HybridRetriever()
