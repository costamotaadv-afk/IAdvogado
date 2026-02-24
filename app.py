import streamlit as st
import os
from src.pdf_processor import extract_text_from_pdf, split_text_into_chunks
from src.rag_manager import RAGManager
from src.web_search import search_jurisprudence
from src.opinion_generator import generate_legal_opinion

st.set_page_config(
    page_title="Assistente de Pareceres Jurídicos - Lei 14.133/2021",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Assistente de Pareceres Jurídicos - Licitações e Contratos")
st.markdown("""
Este aplicativo auxilia advogados publicistas na elaboração de pareceres jurídicos com base na **Lei Federal nº 14.133/2021**, 
jurisprudência do TCU, STJ, STF e Tribunais de Justiça.
""")

# Inicializa o RAG Manager
@st.cache_resource
def get_rag_manager(api_key=None):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    return RAGManager()

# Sidebar para configurações e upload de base de conhecimento
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        
    st.divider()

# Inicializa o RAG Manager apenas se a chave da API estiver presente
rag_manager = None
if api_key:
    rag_manager = get_rag_manager(api_key)
    
    st.header("📚 Base de Conhecimento (RAG)")
    st.markdown("Faça upload de PDFs com a Lei 14.133, Manuais do TCU, etc., para alimentar a base de dados local.")
    knowledge_files = st.file_uploader("Upload de Leis/Manuais (PDF)", type="pdf", accept_multiple_files=True, key="knowledge")
    
    if st.button("Processar Base de Conhecimento"):
        if not api_key:
            st.error("Insira a chave da OpenAI primeiro.")
        elif knowledge_files:
            with st.spinner("Processando documentos e criando embeddings..."):
                for file in knowledge_files:
                    text = extract_text_from_pdf(file)
                    chunks = split_text_into_chunks(text)
                    rag_manager.add_documents(chunks, [{"source": file.name}] * len(chunks))
                st.success("Base de conhecimento atualizada com sucesso!")
        else:
            st.warning("Nenhum arquivo selecionado.")

# Área principal para análise de caso
st.header("📄 Análise de Caso Concreto")
case_file = st.file_uploader("Faça upload do documento a ser analisado (Edital, Contrato, Processo)", type="pdf", key="case")
query_topic = st.text_input("Qual o tema principal da análise? (Ex: Dispensa de licitação por valor, inexigibilidade, reequilíbrio econômico-financeiro)")

if st.button("Gerar Parecer Jurídico", type="primary"):
    if not api_key:
        st.error("Por favor, insira sua OpenAI API Key na barra lateral.")
    elif not case_file:
        st.error("Por favor, faça o upload do documento a ser analisado.")
    elif not query_topic:
        st.error("Por favor, informe o tema principal da análise.")
    else:
        with st.status("Processando sua solicitação...", expanded=True) as status:
            st.write("📄 Extraindo texto do documento...")
            case_text = extract_text_from_pdf(case_file)
            
            st.write("🔍 Buscando na base de conhecimento local (RAG)...")
            rag_results = rag_manager.search_similar(query_topic, k=5)
            rag_context = "\n\n".join([doc.page_content for doc in rag_results]) if rag_results else "Nenhum contexto local encontrado."
            
            st.write("🌐 Buscando jurisprudência recente na Web (TCU, STJ, TJs)...")
            search_query = f"jurisprudência TCU STJ TJ {query_topic} lei 14.133"
            web_context = search_jurisprudence(search_query)
            
            st.write("✍️ Gerando Parecer Jurídico com IA...")
            status.update(label="Parecer gerado com sucesso!", state="complete", expanded=False)
            
        try:
            st.markdown("### 📝 Parecer Jurídico")
            
            # Container para o streaming do texto
            response_container = st.empty()
            full_response = ""
            
            # Chama a função que agora retorna um stream
            stream = generate_legal_opinion(
                pdf_text=case_text,
                rag_context=rag_context,
                web_context=web_context
            )
            
            # Itera sobre os chunks do stream e atualiza a interface
            for chunk in stream:
                if hasattr(chunk, "content"):
                    full_response += chunk.content
                else:
                    full_response += str(chunk)
                response_container.markdown(full_response + "▌")
            
            # Remove o cursor piscante no final
            response_container.markdown(full_response)
            
            # Expander para mostrar o contexto utilizado
            with st.expander("Ver Contexto Utilizado (RAG e Web)"):
                st.markdown("#### Contexto Local (RAG)")
                if rag_results:
                    for i, doc in enumerate(rag_results):
                        source = doc.metadata.get("source", "Desconhecida")
                        st.info(f"**Fonte {i+1}: {source}**\n\n{doc.page_content}")
                else:
                    st.write("Nenhum contexto local encontrado.")
                    
                st.markdown("#### Jurisprudência Web")
                st.info(web_context)
            
            # Opção para download
            st.download_button(
                label="Baixar Parecer (TXT)",
                data=full_response,
                file_name="parecer_juridico.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Erro ao gerar o parecer: {str(e)}")
