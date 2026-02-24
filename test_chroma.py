from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Substitua por sua chave real
api_key = "SUA_CHAVE_OPENAI"
persist_directory = "chroma_db"

embeddings = OpenAIEmbeddings(api_key=api_key)
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)
print("Chroma inicializado com sucesso!")
