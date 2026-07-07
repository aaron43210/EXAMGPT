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

    def retrieve(self, query: str, top_k: int = 4, doc_type_filter: str = None, course_id_filter: int = None) -> list[Document]:
        """Retrieve documents with optional doc_type and course_id filtering."""
        if not self.documents:
            return []

        # Get enough candidates to survive filtering
        search_k = top_k * 10
        bm25_indices, _ = self.bm25_retriever.search(query, search_k)
        faiss_indices, _ = self.faiss_retriever.search(query, search_k)

        fused_scores = {}
        for rank, doc_idx in enumerate(bm25_indices):
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + 1 / (rank + 60)

        for rank, doc_idx in enumerate(faiss_indices):
            if doc_idx != -1 and doc_idx < len(self.documents):
                fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + 1 / (rank + 60)

        sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        results = []
        for idx in sorted_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Apply filters
                if doc_type_filter and doc.metadata.get("doc_type", "notes") != doc_type_filter:
                    continue
                if course_id_filter is not None and doc.metadata.get("course_id") != course_id_filter:
                    continue
                    
                results.append(doc)
                if len(results) >= top_k:
                    break
                    
        return results


# Global instance
hybrid_retriever = HybridRetriever()
