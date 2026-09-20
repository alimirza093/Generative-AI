import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📘", layout="wide")


def get_temp_upload_dir():
    if "temp_upload_dir" not in st.session_state:
        st.session_state.temp_upload_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    return st.session_state.temp_upload_dir


def inject_animated_background():
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #020817;
            --bg-2: #0f172a;
            --panel: rgba(15, 23, 42, 0.72);
            --panel-strong: rgba(15, 23, 42, 0.9);
            --border: rgba(148, 163, 184, 0.18);
            --text: #e2e8f0;
            --muted: #94a3b8;
            --accent: #8b5cf6;
            --accent-2: #22d3ee;
            --success: #34d399;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, var(--bg-1), var(--bg-2), #111827);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(139, 92, 246, 0.22), transparent 24%),
                radial-gradient(circle at 82% 12%, rgba(34, 211, 238, 0.15), transparent 32%),
                radial-gradient(circle at 55% 80%, rgba(59, 130, 246, 0.15), transparent 28%),
                linear-gradient(135deg, var(--bg-1), var(--bg-2), #111827);
            background-size: 180% 180%;
            animation: bgShift 18s ease infinite;
        }

        @keyframes bgShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .block-container {
            padding-top: 2.0rem;
            padding-bottom: 2.5rem;
        }

        div[data-testid="stSidebar"] {
            background: rgba(2, 6, 23, 0.75);
            border-right: 1px solid var(--border);
        }

        .glass-panel {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 20px 35px rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(10px);
            padding: 1rem 1.2rem;
        }

        .chat-message {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
        }

        .stChatMessage {
            background: transparent;
        }

        .title-glow {
            color: #f8fafc;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.2rem;
            text-shadow: 0 0 18px rgba(139, 92, 246, 0.35);
        }

        .subtitle {
            color: var(--muted);
            margin-top: 0;
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_animated_background()


def build_vector_db(pdf_path: Path, persist_dir: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    if not docs:
        raise ValueError("The uploaded PDF could not be read or has no pages.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embed_model = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=str(persist_dir),
    )
    return vectorstore, embed_model


def initialize_state():
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = MistralAIEmbeddings(model="mistral-embed")

    if "llm_model" not in st.session_state:
        st.session_state.llm_model = ChatMistralAI(model="open-mistral-7b", temperature=0.7)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None


initialize_state()

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI assistant.

Use ONLY the provided context. If the answer is not present in the context, reply exactly:
"I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)

st.markdown(
    """
    <div style="padding-bottom: 1rem;">
        <div class="title-glow">PDF RAG Assistant</div>
        <p class="subtitle">Ask questions about the uploaded PDF and get grounded answers from the indexed document.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("Document control")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("Build / Refresh DB", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a PDF first. Answers are generated only from the uploaded document.")
        else:
            upload_dir = get_temp_upload_dir()
            source_pdf = upload_dir / uploaded_file.name
            source_pdf.write_bytes(uploaded_file.getvalue())

            db_dir = upload_dir / "chroma_db"

            try:
                with st.spinner("Indexing the uploaded document..."):
                    vectorstore, embed_model = build_vector_db(source_pdf, db_dir)

                st.session_state.vectorstore = vectorstore
                st.session_state.embed_model = embed_model
                st.session_state.retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                )
                st.session_state.current_pdf = source_pdf.name
                st.success(f"Database built from uploaded file: {source_pdf.name}")
            except Exception as exc:
                st.error(f"Failed to create the vector DB: {exc}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

    if "current_pdf" in st.session_state:
        st.caption(f"Indexed file: {st.session_state.current_pdf}")
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.get("retriever") is None:
    st.info("Upload a PDF and click Build / Refresh DB to index it before asking questions.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching for relevant passages..."):
        documents = st.session_state.retriever.invoke(prompt)
        context = "\n\n".join(doc.page_content for doc in documents)

    final_prompt = prompt_template.invoke({"context": context, "question": prompt})
    response = st.session_state.llm_model.invoke(final_prompt)
    answer = response.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
