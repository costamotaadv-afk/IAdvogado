import streamlit as st
import os
from src.document_processor import extract_text_from_file, split_text_into_chunks
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
        
    selected_model = st.selectbox(
        "Modelo de IA",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="Modelos 'mini' ou 'turbo' são mais rápidos e baratos. O 'gpt-4o' é mais avançado e preciso."
    )
        
    st.divider()

# Define os tipos de arquivos permitidos globalmente para ser usado em todo o app
allowed_types = ["pdf", "docx", "txt", "rtf", "xlsx", "xls", "csv", "png", "jpeg", "jpg", "webp", "heic"]

# Inicializa o RAG Manager apenas se a chave da API estiver presente
rag_manager = None

if api_key:
    rag_manager = get_rag_manager(api_key)
    
    st.info("💡 Dica: Você pode gerenciar e pesquisar seus documentos na aba 'Biblioteca de Documentos'.")

# Criação de abas para separar as funcionalidades
tab1, tab2 = st.tabs(["📄 Gerador de Pareceres", "📚 Biblioteca de Documentos"])

with tab1:
    # Área principal para análise de caso
    st.header("📄 Análise de Caso Concreto")
    st.markdown("Escolha o documento que será o **objeto principal da análise** (você pode fazer upload de um novo, escolher da biblioteca, ou ambos):")
    
    col_a, col_b = st.columns(2)
    with col_a:
        case_file = st.file_uploader(
            "1. Fazer upload de novo documento", 
            type=allowed_types, 
            key="case"
        )
    with col_b:
        available_sources = rag_manager.get_all_sources() if rag_manager else []
        selected_lib_docs = st.multiselect(
            "2. Selecionar da Biblioteca",
            options=available_sources,
            help="Selecione um ou mais documentos já salvos na sua biblioteca para serem analisados."
        )

    query_topic = st.text_input("Qual o tema principal da análise? (Ex: Dispensa de licitação por valor, inexigibilidade, reequilíbrio econômico-financeiro)")
    
    use_library = st.toggle(
        "📚 Usar a Biblioteca como base de pesquisa (Contexto)", 
        value=True,
        help="Se ativado, a IA vai pesquisar na sua Biblioteca de Documentos para fundamentar o parecer (similar ao NotebookLM)."
    )

    if st.button("Gerar Parecer Jurídico", type="primary"):
        if not api_key:
            st.error("Por favor, insira sua OpenAI API Key na barra lateral.")
        elif not case_file and not selected_lib_docs:
            st.error("Por favor, faça o upload de um documento ou selecione um da biblioteca para ser analisado.")
        elif not query_topic:
            st.error("Por favor, informe o tema principal da análise.")
        else:
            with st.status("Processando sua solicitação...", expanded=True) as status:
                st.write("📄 Extraindo texto do(s) documento(s) alvo...")
                case_text = ""
                
                if case_file:
                    try:
                        case_text += f"--- Documento Novo: {case_file.name} ---\n"
                        case_text += extract_text_from_file(case_file, case_file.name) + "\n\n"
                    except Exception as e:
                        st.error(f"Erro ao extrair texto do upload: {str(e)}")
                        st.stop()
                        
                if selected_lib_docs and rag_manager:
                    for doc_source in selected_lib_docs:
                        case_text += f"--- Documento da Biblioteca: {doc_source} ---\n"
                        case_text += rag_manager.get_text_by_source(doc_source) + "\n\n"
                
                rag_context = "O usuário optou por não utilizar a Biblioteca de Documentos."
                rag_results = []
                
                if use_library:
                    st.write("🔍 Buscando na base de conhecimento local (Biblioteca)...")
                    rag_results = rag_manager.search_similar(query_topic, k=5)
                    rag_context = "\n\n".join([doc.page_content for doc in rag_results]) if rag_results else "Nenhum contexto local encontrado na Biblioteca."
                
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
                    web_context=web_context,
                    model_name=selected_model
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
                with st.expander("Ver Contexto Utilizado (Biblioteca e Web)"):
                    st.markdown("#### Contexto Local (Biblioteca)")
                    if not use_library:
                        st.write("A opção de usar a Biblioteca estava desativada.")
                    elif rag_results:
                        for i, doc in enumerate(rag_results):
                            source = doc.metadata.get("source", "Desconhecida")
                            st.info(f"**Fonte {i+1}: {source}**\n\n{doc.page_content}")
                    else:
                        st.write("Nenhum contexto local encontrado na Biblioteca.")
                        
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

with tab2:
    st.header("� Gerenciar Biblioteca de Documentos")
    st.markdown("Faça upload de documentos (Leis, Manuais, Jurisprudência) para alimentar a base de dados local.")
    
    # Área de Upload na aba da Biblioteca
    lib_knowledge_files = st.file_uploader(
        "Upload de Documentos (Até 500MB)", 
        type=allowed_types, 
        accept_multiple_files=True, 
        key="lib_knowledge",
        help="Recomendamos enviar os arquivos aos poucos (cumulativamente) para não sobrecarregar o processamento."
    )
    
    if st.button("Processar Documentos para a Biblioteca", key="btn_process_lib"):
        if not api_key:
            st.error("Insira a chave da OpenAI na barra lateral primeiro.")
        elif lib_knowledge_files:
            with st.spinner("Processando documentos e criando embeddings..."):
                for file in lib_knowledge_files:
                    try:
                        text = extract_text_from_file(file, file.name)
                        if text.strip():
                            chunks = split_text_into_chunks(text)
                            rag_manager.add_documents(chunks, [{"source": file.name}] * len(chunks))
                            st.success(f"✅ {file.name} processado com sucesso!")
                        else:
                            st.warning(f"⚠️ Nenhum texto extraído de {file.name}.")
                    except Exception as e:
                        st.error(f"❌ Erro ao processar {file.name}: {str(e)}")
                st.success("🎉 Base de conhecimento atualizada com sucesso!")
                st.rerun()
        else:
            st.warning("Nenhum arquivo selecionado.")

    st.divider()
    
    st.header("📂 Documentos Anexados")
    if rag_manager:
        sources = rag_manager.get_all_sources()
        if sources:
            st.write(f"Você tem **{len(sources)}** documento(s) na sua biblioteca:")
            for source in sources:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 {source}")
                with col2:
                    if st.button("🗑️ Excluir", key=f"del_{source}"):
                        rag_manager.delete_by_source(source)
                        st.success(f"Documento '{source}' excluído com sucesso!")
                        st.rerun()
        else:
            st.info("Sua biblioteca está vazia. Faça o upload de documentos acima.")
    else:
        st.warning("Insira a chave da OpenAI na barra lateral para ver seus documentos.")

    st.divider()

    st.header("🔍 Pesquisa na Biblioteca de Documentos")
    st.markdown("Pesquise diretamente nos documentos que você fez upload na base de conhecimento.")
    
    lib_search_query = st.text_input("Digite o termo ou assunto que deseja pesquisar:", key="library_search")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        num_results = st.number_input("Número de resultados", min_value=1, max_value=20, value=5)
    
    if st.button("Pesquisar na Biblioteca", type="primary", key="btn_lib_search"):
        if not api_key:
            st.error("Por favor, insira sua OpenAI API Key na barra lateral.")
        elif not lib_search_query:
            st.warning("Digite um termo para pesquisar.")
        elif not rag_manager:
            st.error("O gerenciador de base de dados não foi inicializado.")
        else:
            with st.spinner("Buscando nos documentos..."):
                results = rag_manager.search_similar(lib_search_query, k=num_results)
                
                if results:
                    st.success(f"Encontrados {len(results)} trechos relevantes na sua base de dados.")
                    for i, doc in enumerate(results):
                        source = doc.metadata.get("source", "Desconhecida")
                        with st.expander(f"Resultado {i+1} - Documento: {source}", expanded=(i==0)):
                            st.markdown(f"**Fonte:** `{source}`")
                            st.markdown("**Trecho exato encontrado no documento:**")
                            st.info(doc.page_content)
                else:
                    st.info("Nenhum resultado encontrado para esta pesquisa. Tente fazer o upload de mais documentos na barra lateral.")
