import os
from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class RAGManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Inicializa o gerenciador do banco de dados vetorial (RAG).
        
        Args:
            persist_directory (str): Diretório onde o banco de dados será salvo.
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self._initialize_db()

    def _initialize_db(self):
        """Inicializa ou carrega o banco de dados vetorial."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """
        Adiciona documentos ao banco de dados vetorial.
        
        Args:
            texts (List[str]): Lista de textos para adicionar.
            metadatas (List[dict], optional): Metadados associados aos textos.
        """
        if not texts:
            return
            
        documents = [
            Document(page_content=text, metadata=metadatas[i] if metadatas else {})
            for i, text in enumerate(texts)
        ]
        
        self.vectorstore.add_documents(documents)

    def search_similar(self, query: str, k: int = 4) -> List[Document]:
        """
        Busca documentos similares no banco de dados.
        
        Args:
            query (str): A consulta de busca.
            k (int): Número de resultados a retornar.
            
        Returns:
            List[Document]: Lista de documentos similares encontrados.
        """
        if not self.vectorstore:
            return []
            
        return self.vectorstore.similarity_search(query, k=k)

    def get_all_sources(self) -> List[str]:
        """
        Retorna uma lista com os nomes de todos os arquivos (sources) únicos no banco de dados.
        """
        if not self.vectorstore:
            return []
        try:
            # Acessa a coleção subjacente do ChromaDB
            collection = self.vectorstore._collection
            result = collection.get(include=["metadatas"])
            metadatas = result.get("metadatas", [])
            
            sources = set()
            for meta in metadatas:
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sorted(list(sources))
        except Exception as e:
            print(f"Erro ao obter fontes: {e}")
            return []

    def get_text_by_source(self, source: str) -> str:
        """
        Recupera todo o texto associado a uma fonte específica.
        """
        if not self.vectorstore:
            return ""
        try:
            collection = self.vectorstore._collection
            result = collection.get(where={"source": source}, include=["documents"])
            documents = result.get("documents", [])
            return "\n\n".join(documents)
        except Exception as e:
            print(f"Erro ao obter texto da fonte {source}: {e}")
            return ""

    def delete_by_source(self, source: str):
        """
        Remove todos os documentos associados a uma fonte específica.
        """
        if not self.vectorstore:
            return
        try:
            collection = self.vectorstore._collection
            collection.delete(where={"source": source})
        except Exception as e:
            print(f"Erro ao deletar fonte {source}: {e}")
