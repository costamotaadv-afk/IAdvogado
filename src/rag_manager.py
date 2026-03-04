import os
from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.document_processor import extract_text_from_file, split_text_into_chunks

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
        self.vector_store = None
        self._initialize_db()
        self.load_legal_knowledge_base()

    def _initialize_db(self):
        """Inicializa ou carrega o banco de dados vetorial."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self.vector_store = self.vectorstore

    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """
        Adiciona documentos ao banco de dados vetorial.
        
        Args:
            texts (List[str]): Lista de textos para adicionar.
            metadatas (List[dict], optional): Metadados associados aos textos.
        """
        if not texts:
            return

        if not self.vectorstore:
            self._initialize_db()

        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(page_content=text, metadata=metadata))

        self.vectorstore.add_documents(documents)
        if hasattr(self.vectorstore, "persist"):
            self.vectorstore.persist()

    def load_legal_knowledge_base(self):
        """
        Carrega automaticamente a base jurídica fixa
        (Lei 14.133, STF, STJ, TCU, princípios etc.)
        """

        base_path = os.path.join(os.path.dirname(__file__), "knowledge_base")

        allowed_types = (".pdf", ".doc", ".docx", ".txt")

        for root, _, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(allowed_types):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "rb") as f:
                            text = extract_text_from_file(f, file)

                        if text.strip():
                            chunks = split_text_into_chunks(text)
                            metadata = [{"source": file}] * len(chunks)
                            self.add_documents(chunks, metadata)
                            print(f"Base jurídica carregada: {file}")
                    except Exception as e:
                        print(f"Erro ao carregar {file}: {str(e)}")

    def search_similar(self, query: str, k: int = 4) -> List[Document]:
        """
        Busca documentos similares no banco de dados.
        
        Args:
            query (str): A consulta de busca.
            k (int): Número de resultados a retornar.
            
        Returns:
            List[Document]: Lista de documentos similares encontrados.
        """
        if not self.vector_store:
            return []

        results = self.vector_store.similarity_search(query, k=8)
        return results

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
