"""
Script de exemplo para testar a inicialização do ChromaDB.
Este não é um teste automatizado. Execute manualmente quando necessário.

Para usar este script:
1. Configure a variável de ambiente OPENAI_API_KEY
2. Execute: python test_chroma.py
"""
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def main():
    # Busca a chave da variável de ambiente
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("ERRO: OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        print("Configure-a antes de executar este script.")
        return
    
    persist_directory = "chroma_db"
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("Chroma inicializado com sucesso!")

if __name__ == "__main__":
    main()
