import logging
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

logger = logging.getLogger(__name__)


class RAGManager:
    """Manages a local RAG (Retrieval-Augmented Generation) vector store."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    def add_documents(self, chunks: list[str], metadatas: list[dict] | None = None) -> None:
        """Add text chunks to the vector store."""
        if not chunks:
            return
        if metadatas is None:
            metadatas = [{}] * len(chunks)
        documents = [
            Document(page_content=chunk, metadata=meta)
            for chunk, meta in zip(chunks, metadatas)
        ]
        self.vectorstore.add_documents(documents)

    def search_similar(self, query: str, k: int = 5) -> list[Document]:
        """Search for documents similar to the query."""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
        except Exception as exc:
            logger.warning("Vector store similarity search failed: %s", exc)
            results = []
        return results
