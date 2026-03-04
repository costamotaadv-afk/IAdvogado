import streamlit as st
import os
import time
from dotenv import load_dotenv
from src.document_processor import extract_text_from_file, split_text_into_chunks

from src.rag_manager import RAGManager
from src.web_search import search_jurisprudence
from src.opinion_generator import generate_legal_opinion, get_contextual_analysis_report, get_comprehensive_linguistic_analysis

# Carrega as variáveis de ambiente do arquivo .env (se existir)
load_dotenv()

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
    if api_key and api_key.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = api_key
    return RAGManager()

# Sidebar para configurações e upload de base de conhecimento
with st.sidebar:
    st.header("⚙️ Configurações")

    # 1. Tenta pegar de st.secrets (Streamlit Cloud)
    try:
        secret_key = st.secrets["OPENAI_API_KEY"]
    except (FileNotFoundError, KeyError, Exception):
        secret_key = ""

    # 2. Se não achou, tenta variável de ambiente local
    if not secret_key:
        secret_key = os.getenv("OPENAI_API_KEY", "")

    # Se já tivermos uma chave válida configurada (do Cloud), usamos ela direto
    if secret_key:
        st.success("✅ Chave de API configurada pelo servidor.")
        api_key = secret_key
    else:
        # Só pede a chave se não encontrou em lugar nenhum
        api_key_input = st.text_input("OpenAI API Key", type="password")

        if api_key_input:
            try:
                api_key_input.encode('ascii')
                api_key = api_key_input.strip()
            except UnicodeEncodeError:
                st.error("❌ A chave contém caracteres inválidos.")
                api_key = None
        else:
            api_key = None

    if api_key and api_key.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = api_key

    selected_model = st.selectbox(
        "Modelo de IA",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="""Escolha o modelo:
        • gpt-4o-mini: Rápido e econômico (pareceres curtos)
        • gpt-4o: Avançado e preciso (pareceres médios)
        • gpt-4-turbo: Máxima capacidade (pareceres longos e complexos) ✨ RECOMENDADO
        • gpt-3.5-turbo: Mais econômico (pode truncar pareceres longos)
        """
    )

    # Aviso sobre capacidade do modelo
    if selected_model == "gpt-3.5-turbo":
        st.warning("⚠️ Este modelo pode truncar pareceres longos. Para análises completas, use **gpt-4-turbo**.")
    elif selected_model == "gpt-4-turbo":
        st.success("✅ Modelo ideal para pareceres completos e detalhados!")
    elif selected_model == "gpt-4o":
        st.info("💡 Bom modelo! Para pareceres muito extensos, considere **gpt-4-turbo**.")

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

@st.cache_data(ttl=1800)
def cached_web_search(topic: str) -> str:
    return search_jurisprudence(topic)

with tab1:
    # Área principal para análise de caso
    st.header("📄 Análise de Caso Concreto")
    st.markdown("Faça upload do documento que será o **objeto principal da análise**:")

    case_file = st.file_uploader(
        "Upload de Documento (Edital, Contrato, Processo)",
        type=allowed_types,
        key="case"
    )

    # Variável selected_lib_docs precisa existir mesmo vazia para não quebrar a lógica abaixo
    selected_lib_docs = []
    if rag_manager:
        sources = rag_manager.get_all_sources()
        if sources:
            selected_lib_docs = st.multiselect(
                "Selecione documentos da Biblioteca para análise:",
                options=sources,
                default=[]
            )

    query_topic = st.text_input("Qual o tema principal da análise? (Ex: Dispensa de licitação por valor, inexigibilidade, reequilíbrio econômico-financeiro)")

    use_library = st.toggle(
        "📚 Usar a Biblioteca como base de pesquisa (Contexto)",
        value=True,
        help="Se ativado, a IA vai pesquisar na sua Biblioteca de Documentos para fundamentar o parecer (similar ao NotebookLM)."
    )

    use_web_search = st.toggle(
        "🌐 Buscar na Internet - Motor de Busca",
        value=True,
        help="Se ativado, a IA vai pesquisar na web (Google-like) por legislação atualizada, jurisprudência e doutrina."
    )

    if st.button("Gerar Parecer Jurídico", type="primary"):
        if not api_key:
            st.error("Por favor, insira sua OpenAI API Key na barra lateral.")
        # Se nenhuma opção de entrada fornecida (nem arquivo, nem biblioteca, nem web), aí sim mostra erro
        elif not case_file and not selected_lib_docs and not use_web_search:
            st.error("Por favor, forneça pelo menos uma fonte de informação: Upload de arquivo, Seleção da Biblioteca ou Busca na Web.")
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

                if not case_text:
                    case_text = "Nenhum documento específico foi anexado. O parecer será baseado inteiramente na pesquisa web e conhecimento prévio."

                rag_context = "O usuário optou por não utilizar a Biblioteca de Documentos."
                rag_results = []

                if use_library:
                    st.write("🔍 Buscando na base de conhecimento local (Biblioteca)...")
                    if rag_manager:
                        rag_results = rag_manager.search_similar(query_topic, k=5)
                        if rag_results:
                            rag_limit_chars = 20000
                            rag_parts = []
                            total_chars = 0
                            for doc in rag_results:
                                if total_chars >= rag_limit_chars:
                                    break
                                part = doc.page_content
                                remaining = rag_limit_chars - total_chars
                                rag_parts.append(part[:remaining])
                                total_chars += min(len(part), remaining)
                            rag_context = "\n\n".join(rag_parts)
                        else:
                            rag_context = "Nenhum contexto local relevante encontrado."

                web_context = "O usuário optou por não realizar buscas na Internet."
                if use_web_search:
                    st.write("🌐 Pesquisando Doutrina, Legislação e Jurisprudência na Web (Google)...")
                    web_context = cached_web_search(query_topic)

                # Mostrar análise contextual da solicitação do usuário
                st.write("🧠 Analisando contexto da sua solicitação...")
                contextual_report = get_contextual_analysis_report(query_topic)
                with st.expander("📊 Ver Análise Contextual (Como o agente interpretou sua solicitação)", expanded=False):
                    st.markdown(contextual_report)

                # Mostrar análise linguística completa (aprendizado do agente)
                st.write("📚 Analisando padrões linguísticos e aprendendo com o conteúdo...")
                try:
                    # Garante que o texto não seja muito longo e evita problemas com caracteres especiais
                    analysis_text = query_topic + ' ' + case_text[:5000] if case_text else query_topic
                    linguistic_report = get_comprehensive_linguistic_analysis(analysis_text)
                    with st.expander("🎓 Ver Análise Linguística (Aprendizado do Agente)", expanded=False):
                        st.markdown(linguistic_report)
                        st.info("""
                        💡 **Este agente aprende de verdade!**

                        A cada interação, o agente analisa e memoriza:
                        - Relações entre palavras (vocabulário contextual)
                        - Estrutura de frases (padrões sintáticos)
                        - Padrões de raciocínio (lógica argumentativa)
                        - Formas de argumentar (estratégias persuasivas)
                        - Estilos de escrita (características linguísticas)

                        Quanto mais você usa, mais ele se adapta ao seu estilo e necessidades!
                        """)
                except Exception as e:
                    st.warning(f"⚠️ Análise linguística não disponível: {str(e)}")
                    st.info("O parecer será gerado normalmente, mas sem o relatório de aprendizado linguístico.")

                st.write("✍️ Gerando Parecer Jurídico com IA...")

            output_container = st.container()
            with output_container:
                st.markdown("### 📝 Parecer Jurídico")

                # Container para o streaming do texto (write é mais simples e estável)
                response_container = st.empty()
                full_response = ""
                last_update_time = 0.0
                char_count_since_update = 0

                try:
                    # Chama a função que agora retorna um stream com retry e timeout
                    stream = generate_legal_opinion(
                        pdf_text=case_text,
                        rag_context=rag_context,
                        web_context=web_context,
                        user_query=query_topic,
                        model_name=selected_model
                    )

                    # Itera sobre os chunks do stream e atualiza a interface
                    for chunk in stream:
                        # Stream já retorna strings diretamente após as melhorias
                        full_response += chunk
                        char_count_since_update += len(chunk)
                        current_time = time.time()

                        # Atualiza apenas a cada 300 caracteres OU a cada 0.30 segundos
                        # Reduz a pressão no DOM do Streamlit para evitar removeChild
                        if char_count_since_update >= 300 or (current_time - last_update_time) >= 0.30:
                            try:
                                response_container.write(full_response + "▌")
                                char_count_since_update = 0
                                last_update_time = current_time
                            except Exception:
                                pass

                    # Remove o cursor piscante no final
                    try:
                        response_container.write(full_response)
                    except Exception:
                        # Se falhar, tenta novamente após um delay
                        time.sleep(0.2)
                        response_container.write(full_response)

                    # Se nada foi gerado, alerta o usuário
                    if not full_response.strip():
                        st.error("⚠️ Nenhum texto foi gerado. Tente novamente, selecione gpt-4-turbo e verifique a conexão.")
                    else:
                        # Salva na session_state para preservar o parecer
                        st.session_state['last_opinion'] = full_response
                        status.update(label="Parecer gerado com sucesso!", state="complete", expanded=False)

                except Exception as e:
                    error_message = f"""⚠️ **Erro na geração do parecer:**

                    {str(e)}

                    **Sugestões:**
                    - Tente novamente em alguns instantes
                    - Selecione um modelo menor (gpt-4o-mini)
                    - Verifique se o documento não é muito grande
                    - Verifique sua conexão com a internet
                    """
                    st.error(error_message)
                    full_response = error_message

                # Expander para mostrar o contexto utilizado
                with st.expander("Ver Contexto Utilizado (Biblioteca e Web)"):
                    st.markdown("#### Contexto Local (Biblioteca)")
                    if not use_library:
                        st.write("A opção de usar a Biblioteca estava desativada.")
                    elif rag_results:
                        for i, doc in enumerate(rag_results):
                            source = doc.metadata.get("source", "Desconhecida")
                            st.info(f"**Fonte {i+1}: {source}**")
                            st.text(doc.page_content)
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

with tab2:
    st.header("📚 Gerenciar Biblioteca de Documentos")
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
        elif not rag_manager:
            st.error("O gerenciador da base de dados nao foi inicializado. Verifique a chave da API e tente novamente.")
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
                            st.text(doc.page_content)
                else:
                    st.info("Nenhum resultado encontrado para esta pesquisa. Tente fazer o upload de mais documentos na barra lateral.")
