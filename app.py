import streamlit as st
import os
import time
import importlib
from datetime import datetime
from io import BytesIO
from docx import Document

try:
    canvas = importlib.import_module("reportlab.pdfgen.canvas")
    pagesizes = importlib.import_module("reportlab.lib.pagesizes")
    letter = pagesizes.letter
    HAS_REPORTLAB = True
except Exception:
    canvas = None
    letter = None
    HAS_REPORTLAB = False
from dotenv import load_dotenv
from src.document_processor import extract_text_from_file, split_text_into_chunks

from src.rag_manager import RAGManager
from src.web_search import search_jurisprudence
from src.opinion_generator import generate_legal_opinion, get_contextual_analysis_report, get_comprehensive_linguistic_analysis
from src.legal_classifier import classify_legal_case
from src.template_loader import load_template
from src.process_extractor import extract_process_fields

# Carrega as variáveis de ambiente do arquivo .env (se existir)
load_dotenv()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_opinion" not in st.session_state:
    st.session_state.last_opinion = ""
if "last_contextual_report" not in st.session_state:
    st.session_state.last_contextual_report = ""
if "last_linguistic_report" not in st.session_state:
    st.session_state.last_linguistic_report = ""
if "last_rag_results" not in st.session_state:
    st.session_state.last_rag_results = []
if "last_web_context" not in st.session_state:
    st.session_state.last_web_context = ""

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

# Define os tipos de arquivos permitidos globalmente para ser usado em todo o app
allowed_types = ["pdf", "docx", "txt", "rtf", "xlsx", "xls", "csv", "png", "jpeg", "jpg", "webp", "heic"]

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
        
MAX_CONTEXT_TOKENS = 8000

case_file = None
output_mode = "parecer técnico"
style_examples = ""

MODEL_CONTEXT_LIMITS = {
    "gpt-4o-mini": {"case": 12000, "rag": 8000, "web": 8000, "style": 4000},
    "gpt-4o": {"case": 16000, "rag": 10000, "web": 10000, "style": 6000},
    "gpt-4-turbo": {"case": 12000, "rag": 8000, "web": 8000, "style": 4000},
    "gpt-3.5-turbo": {"case": 8000, "rag": 6000, "web": 6000, "style": 3000},
}


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 4
    return text[:max_chars]


def _is_procurement_doc(file_name: str, text: str) -> bool:
    haystack = f"{file_name} {text}".lower()
    return "edital" in haystack or "contrato" in haystack


def _limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _make_docx_bytes(text: str) -> bytes:
    buffer = BytesIO()
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def _make_pdf_bytes(text: str) -> bytes:
    if not HAS_REPORTLAB:
        raise RuntimeError("Reportlab nao esta instalado.")
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x = 40
    y = height - 40
    line_height = 14

    for line in text.splitlines():
        if y < 40:
            pdf.showPage()
            y = height - 40
        pdf.drawString(x, y, line[:2000])
        y -= line_height

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

# Inicializa o RAG Manager apenas se a chave da API estiver presente
rag_manager = None

if api_key:
    rag_manager = get_rag_manager(api_key)
    
    st.info("💡 Dica: Você pode gerenciar e pesquisar seus documentos na aba 'Biblioteca de Documentos'.")

@st.cache_data(ttl=1800)
def cached_web_search(topic: str) -> str:
    return search_jurisprudence(topic)

col_docs, col_chat = st.columns([1, 2], gap="large")

with col_docs:
    st.header("Biblioteca")
    lib_knowledge_files = st.file_uploader(
        "Adicionar documentos",
        type=allowed_types,
        accept_multiple_files=True,
        key="lib_knowledge",
        help="Recomendamos enviar os arquivos aos poucos (cumulativamente) para nao sobrecarregar o processamento."
    )

    if st.button("Processar Documentos", key="btn_process_lib"):
        if not api_key:
            st.error("Insira a chave da OpenAI na barra lateral primeiro.")
        elif not rag_manager:
            st.error("O gerenciador de base de dados nao foi inicializado. Verifique a chave da API e tente novamente.")
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
                            st.warning(f"⚠️ Nenhum texto extraido de {file.name}.")
                    except Exception as e:
                        st.error(f"❌ Erro ao processar {file.name}: {str(e)}")
                st.success("🎉 Base de conhecimento atualizada com sucesso!")
                st.rerun()
        else:
            st.warning("Nenhum arquivo selecionado.")

    st.divider()

    st.subheader("Documentos carregados")
    if rag_manager:
        sources = rag_manager.get_all_sources()
        if sources:
            for source in sources:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 {source}")
                with col2:
                    if st.button("🗑️", key=f"del_{source}"):
                        rag_manager.delete_by_source(source)
                        st.success(f"Documento '{source}' excluido com sucesso!")
                        st.rerun()
        else:
            st.info("Sua biblioteca esta vazia. Faça o upload de documentos acima.")
    else:
        st.warning("Insira a chave da OpenAI na barra lateral para ver seus documentos.")

    st.divider()

    st.subheader("Pesquisar na biblioteca")
    lib_search_query = st.text_input("Termo ou assunto", key="library_search")
    num_results = st.number_input("Numero de resultados", min_value=1, max_value=20, value=5)

    if st.button("Pesquisar", type="primary", key="btn_lib_search"):
        if not api_key:
            st.error("Por favor, insira sua OpenAI API Key na barra lateral.")
        elif not lib_search_query:
            st.warning("Digite um termo para pesquisar.")
        elif not rag_manager:
            st.error("O gerenciador de base de dados nao foi inicializado.")
        else:
            with st.spinner("Buscando nos documentos..."):
                results = rag_manager.search_similar(lib_search_query, k=num_results)

                if results:
                    for i, doc in enumerate(results):
                        source = doc.metadata.get("source", "Desconhecida")
                        with st.expander(f"Resultado {i+1} - Documento: {source}", expanded=(i == 0)):
                            st.markdown(f"**Fonte:** `{source}`")
                            st.text(doc.page_content)
                else:
                    st.info("Nenhum resultado encontrado.")

with col_chat:
    st.header("Chat Juridico")

    if st.button("Limpar conversa"):
        st.session_state.messages = []
        st.session_state.last_opinion = ""
        st.session_state.last_contextual_report = ""
        st.session_state.last_linguistic_report = ""
        st.session_state.last_rag_results = []
        st.session_state.last_web_context = ""
        st.rerun()

    st.subheader("Documento do caso")
    case_file = st.file_uploader(
        "Upload de Documento (Edital, Contrato, Processo)",
        type=allowed_types,
        key="case"
    )

    output_mode = st.selectbox(
        "Modo de parecer",
        options=["parecer tecnico", "parecer simplificado", "minuta de despacho"],
        index=0,
        help="Selecione o nivel de formalidade e o formato de saida."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Pergunte algo sobre os documentos")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with col_chat:
        with st.chat_message("user"):
            st.markdown(prompt)

    if not api_key:
        error_text = "Por favor, insira sua OpenAI API Key na barra lateral."
        with col_chat:
            with st.chat_message("assistant"):
                st.error(error_text)
        st.session_state.messages.append({"role": "assistant", "content": error_text})
    elif not case_file and not rag_manager:
        error_text = "Por favor, forneça pelo menos uma fonte de informacao: Upload de arquivo ou Biblioteca."
        with col_chat:
            with st.chat_message("assistant"):
                st.error(error_text)
        st.session_state.messages.append({"role": "assistant", "content": error_text})
    else:
        query_topic = prompt
        use_library = True
        use_web_search = True

        with st.spinner("Processando sua solicitacao..."):
            case_type = classify_legal_case(query_topic)
            template_text = load_template(case_type)

            case_text = ""
            if case_file:
                try:
                    case_text += f"--- Documento Novo: {case_file.name} ---\n"
                    case_text += extract_text_from_file(case_file, case_file.name) + "\n\n"
                except Exception as e:
                    error_text = f"Erro ao extrair texto do upload: {str(e)}"
                    with col_chat:
                        with st.chat_message("assistant"):
                            st.error(error_text)
                    st.session_state.messages.append({"role": "assistant", "content": error_text})
                    st.stop()

            if not case_text:
                case_text = "Nenhum documento especifico foi anexado. O parecer sera baseado inteiramente na pesquisa web e conhecimento previo."

            structured_fields = {}
            if case_file and _is_procurement_doc(case_file.name, case_text):
                structured_fields = extract_process_fields(case_text)
                if any(structured_fields.values()):
                    structured_block = "DADOS ESTRUTURADOS DO PROCESSO:\n"
                    for key, value in structured_fields.items():
                        if value:
                            structured_block += f"- {key}: {value}\n"
                    case_text = structured_block + "\n" + case_text

            rag_context = "Nenhum contexto local relevante encontrado."
            rag_results = []
            if use_library and rag_manager:
                rag_results = rag_manager.search_similar(query_topic, k=5)
                if rag_results:
                    rag_limit_tokens = MAX_CONTEXT_TOKENS
                    rag_parts = []
                    total_tokens = 0
                    for doc in rag_results:
                        if total_tokens >= rag_limit_tokens:
                            break
                        part = doc.page_content
                        part_tokens = _approx_token_count(part)
                        remaining_tokens = rag_limit_tokens - total_tokens
                        if part_tokens > remaining_tokens:
                            part = _trim_to_tokens(part, remaining_tokens)
                            part_tokens = _approx_token_count(part)
                        rag_parts.append(part)
                        total_tokens += part_tokens
                    rag_context = "\n\n".join(rag_parts)

            web_context = cached_web_search(query_topic) if use_web_search else ""

            limits = MODEL_CONTEXT_LIMITS.get(selected_model, MODEL_CONTEXT_LIMITS["gpt-4o-mini"])
            case_text = _limit_text(case_text, limits["case"])
            rag_context = _limit_text(rag_context, limits["rag"])
            web_context = _limit_text(web_context, limits["web"])
            style_examples_limited = _limit_text(style_examples, limits["style"])

            contextual_report = get_contextual_analysis_report(query_topic)
            st.session_state.last_contextual_report = contextual_report

            try:
                analysis_text = query_topic + " " + case_text[:5000] if case_text else query_topic
                linguistic_report = get_comprehensive_linguistic_analysis(analysis_text)
            except Exception:
                linguistic_report = ""
            st.session_state.last_linguistic_report = linguistic_report
            st.session_state.last_rag_results = rag_results
            st.session_state.last_web_context = web_context

        with col_chat:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response_state = {
                    "full_response": "",
                    "last_update_time": 0.0,
                    "char_count_since_update": 0,
                }

                try:
                    def stream_and_append(chapter_label: str, chapter_key: str) -> None:
                        if chapter_label:
                            response_state["full_response"] += f"\n\n### {chapter_label}\n"

                        stream = generate_legal_opinion(
                            pdf_text=case_text,
                            rag_context=rag_context,
                            web_context=web_context,
                            user_query=query_topic,
                            model_name=selected_model,
                            template_text=template_text,
                            output_mode=output_mode,
                            style_examples=style_examples_limited,
                            chapter=chapter_key
                        )

                        for chunk in stream:
                            response_state["full_response"] += chunk
                            response_state["char_count_since_update"] += len(chunk)
                            current_time = time.time()

                            if (
                                response_state["char_count_since_update"] >= 300
                                or (current_time - response_state["last_update_time"]) >= 0.30
                            ):
                                message_placeholder.markdown(response_state["full_response"] + "▌")
                                response_state["char_count_since_update"] = 0
                                response_state["last_update_time"] = current_time

                    stream_and_append("", "completo")

                    message_placeholder.markdown(response_state["full_response"])

                    if not response_state["full_response"].strip():
                        response_state["full_response"] = "⚠️ Nenhum texto foi gerado. Tente novamente e verifique a conexao."
                        message_placeholder.markdown(response_state["full_response"])

                except Exception as e:
                    response_state["full_response"] = f"⚠️ **Erro na geracao do parecer:**\n\n{str(e)}"
                    message_placeholder.markdown(response_state["full_response"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_state["full_response"]
        })
        st.session_state.last_opinion = response_state["full_response"]

with col_chat:
    if st.session_state.last_contextual_report:
        with st.expander("Ver Analise Contextual"):
            st.markdown(st.session_state.last_contextual_report)

    if st.session_state.last_linguistic_report:
        with st.expander("Ver Analise Linguistica"):
            st.markdown(st.session_state.last_linguistic_report)

    if st.session_state.last_rag_results or st.session_state.last_web_context:
        with st.expander("Ver Contexto Utilizado (Biblioteca e Web)"):
            st.markdown("#### Contexto Local (Biblioteca)")
            if st.session_state.last_rag_results:
                for i, doc in enumerate(st.session_state.last_rag_results):
                    source = doc.metadata.get("source", "Desconhecida")
                    st.info(f"**Fonte {i+1}: {source}**")
                    st.text(doc.page_content)
            else:
                st.write("Nenhum contexto local encontrado na Biblioteca.")

            st.markdown("#### Jurisprudencia Web")
            if st.session_state.last_web_context:
                st.info(st.session_state.last_web_context)

    if st.session_state.last_opinion:
        col_txt, col_docx, col_pdf = st.columns(3)
        with col_txt:
            st.download_button(
                label="Baixar Parecer (TXT)",
                data=st.session_state.last_opinion,
                file_name="parecer_juridico.txt",
                mime="text/plain"
            )
        with col_docx:
            st.download_button(
                label="Baixar Parecer (DOCX)",
                data=_make_docx_bytes(st.session_state.last_opinion),
                file_name="parecer_juridico.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        with col_pdf:
            if HAS_REPORTLAB:
                st.download_button(
                    label="Baixar Parecer (PDF)",
                    data=_make_pdf_bytes(st.session_state.last_opinion),
                    file_name="parecer_juridico.pdf",
                    mime="application/pdf"
                )
            else:
                st.caption("PDF indisponivel: instale reportlab.")
