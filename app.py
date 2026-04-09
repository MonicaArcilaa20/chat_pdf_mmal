import os
import io
import html
import hashlib
import platform

import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


st.set_page_config(
    page_title="RAG para PDFs",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
            :root {
                --primary: #4F46E5;
                --primary-2: #7C3AED;
                --bg: #F4F7FB;
                --card: #FFFFFF;
                --text: #0F172A;
                --muted: #475569;
                --border: #E2E8F0;
            }

            .stApp {
                background: linear-gradient(180deg, #F8FAFC 0%, #F4F7FB 60%, #EEF2FF 100%);
            }

            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1180px;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
            }

            [data-testid="stSidebar"] * {
                color: #F8FAFC;
            }

            .hero-card {
                padding: 1.7rem;
                border: 1px solid rgba(79, 70, 229, 0.14);
                border-radius: 24px;
                background: linear-gradient(
                    135deg,
                    rgba(79, 70, 229, 0.10),
                    rgba(124, 58, 237, 0.08),
                    rgba(255, 255, 255, 0.92)
                );
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            }

            .hero-title {
                font-size: 2.25rem;
                font-weight: 800;
                line-height: 1.1;
                color: var(--text);
                margin-bottom: 0.6rem;
            }

            .hero-subtitle {
                font-size: 1rem;
                color: var(--muted);
                line-height: 1.65;
                margin-bottom: 1rem;
            }

            .badge {
                display: inline-block;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                background: rgba(79, 70, 229, 0.10);
                color: var(--primary);
                font-weight: 700;
                font-size: 0.83rem;
                margin-right: 0.45rem;
                margin-bottom: 0.45rem;
                border: 1px solid rgba(79, 70, 229, 0.15);
            }

            .step-card {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 1rem 1.05rem;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
                margin-bottom: 1rem;
            }

            .step-title {
                font-size: 1rem;
                font-weight: 800;
                color: var(--text);
                margin-bottom: 0.25rem;
            }

            .step-text {
                font-size: 0.95rem;
                color: var(--muted);
                line-height: 1.55;
                margin: 0;
            }

            [data-testid="metric-container"] {
                background: rgba(255, 255, 255, 0.98);
                border: 1px solid var(--border);
                padding: 0.95rem 1rem;
                border-radius: 18px;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
            }

            div.stButton > button,
            div.stFormSubmitButton > button {
                width: 100%;
                border-radius: 14px;
                border: 0;
                height: 3rem;
                font-weight: 700;
                background: linear-gradient(135deg, var(--primary), var(--primary-2));
                color: white;
                box-shadow: 0 10px 24px rgba(79, 70, 229, 0.25);
            }

            div.stButton > button:hover,
            div.stFormSubmitButton > button:hover {
                filter: brightness(1.04);
            }

            [data-testid="stFileUploader"] {
                background: rgba(255, 255, 255, 0.98);
                border: 1px dashed #C7D2FE;
                border-radius: 18px;
                padding: 0.7rem;
            }

            [data-testid="stTextInputRootElement"],
            [data-testid="stTextAreaRootElement"] {
                background: rgba(255, 255, 255, 0.98);
                border-radius: 14px;
            }

            .answer-card {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-left: 6px solid var(--primary);
                border-radius: 18px;
                padding: 1.1rem 1.25rem;
                box-shadow: 0 12px 26px rgba(15, 23, 42, 0.05);
            }

            .answer-card h3 {
                margin: 0 0 0.55rem 0;
                color: var(--text);
            }

            .answer-card p {
                color: var(--muted);
                line-height: 1.75;
                margin: 0;
            }

            .small-note {
                color: var(--muted);
                font-size: 0.90rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    defaults = {
        "knowledge_base": None,
        "pdf_hash": None,
        "file_name": None,
        "total_pages": 0,
        "char_count": 0,
        "chunk_count": 0,
        "preview_text": "",
        "last_answer": "",
        "last_docs": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_bytes: bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages_text).strip()
    return text, len(reader.pages)


@st.cache_data(show_spinner=False)
def split_document(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=120,
        length_function=len,
    )
    return splitter.split_text(text)


def build_vector_store(chunks, api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_texts(chunks, embeddings)


def ask_pdf(question: str, model_name: str, api_key: str, k: int):
    os.environ["OPENAI_API_KEY"] = api_key
    docs = st.session_state.knowledge_base.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model=model_name, temperature=0)

    messages = [
        (
            "system",
            "Eres un asistente experto en análisis documental. "
            "Responde únicamente con base en el contexto recuperado del PDF. "
            "Si la respuesta no está en el contexto, dilo claramente. "
            "Responde en español claro, preciso y útil."
        ),
        (
            "human",
            f"Contexto del documento:\n{context}\n\nPregunta del usuario:\n{question}"
        ),
    ]

    response = llm.invoke(messages)
    return response.content, docs


def render_step_card(title: str, text: str):
    st.markdown(
        f"""
        <div class="step-card">
            <div class="step-title">{html.escape(title)}</div>
            <p class="step-text">{html.escape(text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def reset_app():
    st.session_state.knowledge_base = None
    st.session_state.pdf_hash = None
    st.session_state.file_name = None
    st.session_state.total_pages = 0
    st.session_state.char_count = 0
    st.session_state.chunk_count = 0
    st.session_state.preview_text = ""
    st.session_state.last_answer = ""
    st.session_state.last_docs = []
    st.rerun()


inject_css()
init_state()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    api_key = st.text_input(
        "Clave de OpenAI",
        type="password",
        placeholder="Pega aquí tu clave",
        help="Se usa para generar embeddings y responder preguntas sobre el PDF.",
    )

    model_name = st.text_input(
        "Modelo de respuesta",
        value="gpt-4o-mini",
        help="Puedes cambiarlo si en tu cuenta usas otro modelo.",
    )

    k = st.slider(
        "Fragmentos a recuperar",
        min_value=2,
        max_value=8,
        value=4,
        help="Entre más fragmentos, más contexto; entre menos, más rapidez.",
    )

    st.divider()
    st.markdown("### 🧭 Flujo")
    st.caption("1) Carga el PDF")
    st.caption("2) Procesa el documento")
    st.caption("3) Haz preguntas")
    st.divider()

    st.caption(f"Python: {platform.python_version()}")

    if st.button("Limpiar documento actual"):
        reset_app()

# Header
col_hero, col_image = st.columns([1.45, 1.0], gap="large")

with col_hero:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Analiza PDFs con una interfaz más clara y agradable</div>
            <div class="hero-subtitle">
                Carga un documento, procésalo una vez y luego consulta su contenido con una experiencia
                más ordenada, visual y fácil de usar.
            </div>
            <span class="badge">RAG</span>
            <span class="badge">Streamlit</span>
            <span class="badge">FAISS</span>
            <span class="badge">OpenAI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_image:
    try:
        image = Image.open("Chat_pdf.png")
        st.image(image, width=340)
    except Exception:
        render_step_card(
            "Imagen no disponible",
            "La interfaz funciona normalmente aunque no se encuentre el archivo Chat_pdf.png.",
        )

st.write("")

# Metrics row
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Páginas", st.session_state.total_pages)
with m2:
    st.metric("Fragmentos", st.session_state.chunk_count)
with m3:
    st.metric(
        "Estado",
        "Listo" if st.session_state.knowledge_base is not None else "Pendiente"
    )

if st.session_state.file_name:
    st.caption(f"Documento actual: {st.session_state.file_name}")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📄 Cargar y procesar", "❓ Preguntar", "🔎 Detalles"])

with tab1:
    left, right = st.columns([1.35, 0.9], gap="large")

    with left:
        pdf = st.file_uploader(
            "Carga tu archivo PDF",
            type="pdf",
            help="Sube un documento en PDF para construir la base de conocimiento.",
        )

        process_clicked = st.button(
            "Procesar documento",
            type="primary",
            disabled=not bool(pdf and api_key),
        )

        if not api_key:
            st.info("Primero ingresa tu clave de OpenAI en la barra lateral.")
        elif not pdf:
            st.info("Ahora carga un archivo PDF para habilitar el procesamiento.")

        if process_clicked and pdf and api_key:
            pdf_bytes = pdf.getvalue()
            current_hash = hashlib.md5(pdf_bytes).hexdigest()

            if (
                st.session_state.knowledge_base is not None
                and st.session_state.pdf_hash == current_hash
            ):
                st.success("Ese documento ya fue procesado. Ya puedes hacer preguntas.")
            else:
                try:
                    with st.spinner("Leyendo y vectorizando el documento..."):
                        text, total_pages = extract_text_from_pdf(pdf_bytes)

                        if not text:
                            st.error("No se pudo extraer texto útil del PDF.")
                        else:
                            chunks = split_document(text)
                            knowledge_base = build_vector_store(chunks, api_key)

                            st.session_state.knowledge_base = knowledge_base
                            st.session_state.pdf_hash = current_hash
                            st.session_state.file_name = pdf.name
                            st.session_state.total_pages = total_pages
                            st.session_state.char_count = len(text)
                            st.session_state.chunk_count = len(chunks)
                            st.session_state.preview_text = text[:2500]
                            st.session_state.last_answer = ""
                            st.session_state.last_docs = []

                            st.success("Documento procesado correctamente. Ya puedes consultarlo.")

                except Exception as e:
                    st.error(f"Error al procesar el PDF: {e}")

    with right:
        render_step_card(
            "1. Sube el documento",
            "El usuario primero carga el PDF que quiere analizar."
        )
        render_step_card(
            "2. Procesa una sola vez",
            "Se extrae el texto, se divide en fragmentos y se crea el índice semántico."
        )
        render_step_card(
            "3. Haz preguntas",
            "Después puedes consultar resúmenes, conceptos, conclusiones o datos específicos."
        )

with tab2:
    if st.session_state.knowledge_base is None:
        st.info("Primero procesa un PDF en la pestaña anterior.")
    else:
        st.caption(
            "Ejemplos: “Resume el documento”, “Extrae ideas principales”, "
            "“Dime las conclusiones”, “¿Qué dice sobre X tema?”"
        )

        with st.form("question_form"):
            user_question = st.text_area(
                "Tu pregunta",
                height=120,
                placeholder="Escribe aquí lo que deseas saber del documento...",
            )
            submitted = st.form_submit_button("Consultar documento", type="primary")

        if submitted:
            if not user_question.strip():
                st.warning("Escribe una pregunta antes de consultar.")
            elif not api_key:
                st.warning("Falta la clave de OpenAI en la barra lateral.")
            else:
                try:
                    with st.spinner("Buscando la mejor respuesta..."):
                        answer, docs = ask_pdf(user_question, model_name, api_key, k)
                        st.session_state.last_answer = answer
                        st.session_state.last_docs = docs
                except Exception as e:
                    st.error(f"Error al responder la pregunta: {e}")

        if st.session_state.last_answer:
            safe_answer = html.escape(st.session_state.last_answer).replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="answer-card">
                    <h3>Respuesta</h3>
                    <p>{safe_answer}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Ver fragmentos usados para responder"):
                for i, doc in enumerate(st.session_state.last_docs, start=1):
                    st.markdown(f"**Fragmento {i}**")
                    st.write(doc.page_content[:1200])
                    st.write("---")

with tab3:
    if st.session_state.knowledge_base is None:
        st.info("Cuando proceses un PDF, aquí verás detalles del documento.")
    else:
        d1, d2 = st.columns(2)
        with d1:
            st.write(f"**Nombre del archivo:** {st.session_state.file_name}")
            st.write(f"**Páginas:** {st.session_state.total_pages}")
            st.write(f"**Caracteres extraídos:** {st.session_state.char_count}")
            st.write(f"**Fragmentos generados:** {st.session_state.chunk_count}")

        with d2:
            st.write("**Vista previa del texto extraído**")
            st.text_area(
                "Preview",
                value=st.session_state.preview_text,
                height=240,
                disabled=True,
                label_visibility="collapsed",
            )

        st.markdown(
            '<p class="small-note">Sugerencia: deja el modelo editable en la barra lateral para cambiarlo fácilmente si tu cuenta usa otro nombre de modelo.</p>',
            unsafe_allow_html=True,
        )
