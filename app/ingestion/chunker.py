from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_text(self, text: str, metadata: dict = None) -> list[Document]:
        if metadata is None:
            metadata = {}
        return self.splitter.create_documents([text], metadatas=[metadata])
