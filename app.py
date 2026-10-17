"""Streamlit interface for chatting with a PDF using the existing RAG helpers."""

import hashlib
import html

import fitz
import streamlit as st

from gemini_client import (
    GeminiAuthenticationError,
    GeminiModelError,
    GeminiQuotaError,
    GeminiRequestError,
    GeminiSetupError,
    generate_pdf_answer,
    get_api_key,
    get_model_name,
)
from vector_store import build_faiss_index, chunk_pdf_pages, retrieve_relevant_chunks


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


st.set_page_config(
    page_title="PDF Genius",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the local sentence-transformers model once per Streamlit server."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


st.markdown(
    """
    <style>
    :root {
        --ink: #17283f;
        --muted: #68788d;
        --accent: #3978d4;
        --accent-light: #eaf2ff;
        --surface: #ffffff;
        --surface-soft: #f5f8fc;
        --line: #e4eaf1;
    }
    .stApp { background: #f7f9fc; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247, 249, 252, .92); }
    [data-testid="stSidebar"] {
        background: #f0f4f9;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
    h1, h2, h3, p, label { color: var(--ink); }
    h1 { letter-spacing: -0.04em; font-weight: 750 !important; }
    h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stCaptionContainer"] p, .muted-text { color: var(--muted); }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #aebed1;
        border-radius: 16px;
        background: var(--surface);
    }
    [data-testid="stButton"] button {
        border-radius: 11px;
        min-height: 2.65rem;
        font-weight: 600;
        border-color: #d8e1ec;
        transition: all .15s ease;
    }
    [data-testid="stButton"] button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
    }
    [data-testid="stButton"] button:hover { border-color: var(--accent); color: var(--accent); }
    [data-testid="stButton"] button[kind="primary"]:hover { color: white; background: #2c68bf; }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: .85rem 1rem;
        margin: .7rem 0;
        background: var(--surface);
        box-shadow: 0 4px 16px rgba(23, 40, 63, .035);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--accent-light);
        border-color: #d9e7fc;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 16px;
        border-color: #d7e0eb;
        background: white;
    }
    .brand-mark {
        font-size: 1.12rem; font-weight: 800; letter-spacing: -.025em;
        color: var(--ink); margin: 0 0 .15rem 0;
    }
    .brand-caption { color: var(--muted); font-size: .85rem; margin-bottom: 1.4rem; }
    .welcome-card, .section-card {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 18px; padding: 1.15rem 1.25rem;
        box-shadow: 0 8px 24px rgba(23, 40, 63, .035);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line); border-radius: 16px; background: var(--surface);
    }
    .welcome-card { padding: 1.7rem; margin: .6rem 0 1rem; }
    .welcome-icon {
        width: 3rem; height: 3rem; border-radius: 14px;
        display: inline-flex; align-items: center; justify-content: center;
        color: var(--accent); background: var(--accent-light); font-size: 1.4rem;
        margin-bottom: .9rem;
    }
    .welcome-title { font-size: 1.25rem; font-weight: 720; margin-bottom: .3rem; }
    .welcome-copy { color: var(--muted); line-height: 1.55; }
    .example-card {
        background: white; border: 1px solid var(--line); border-radius: 14px;
        min-height: 6.3rem; padding: .95rem 1rem; color: var(--ink);
        font-size: .92rem; line-height: 1.45;
    }
    .example-label {
        color: var(--muted); font-size: .74rem; text-transform: uppercase;
        font-weight: 700; letter-spacing: .07em; margin-bottom: .45rem;
    }
    .status-pill {
        display: inline-block; border-radius: 999px; padding: .28rem .62rem;
        font-size: .76rem; font-weight: 700; background: #edf1f6; color: #52657b;
    }
    .status-pill.ready { color: #16744a; background: #e6f6ed; }
    .status-pill.processing { color: #8b6412; background: #fff4d9; }
    .page-source {
        display: inline-block; margin: .2rem .35rem .1rem 0; padding: .25rem .55rem;
        border: 1px solid #d8e5f5; border-radius: 999px;
        background: #f2f7ff; color: #365f95; font-size: .78rem;
    }
    @media (max-width: 700px) {
        .welcome-card { padding: 1.2rem; }
        [data-testid="stChatMessage"] { padding: .65rem .75rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


for key, default in {
    "pdf_digest": None,
    "pdf_pages": None,
    "pdf_page_count": 0,
    "pdf_chunks": [],
    "faiss_index": None,
    "embedding_vectors": None,
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


with st.sidebar:
    st.markdown('<div class="brand-mark">📄 PDF Genius</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-caption">Your document workspace</div>', unsafe_allow_html=True)
    st.markdown("### Add a document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Only PDF files are accepted. Scanned PDFs need selectable text; OCR is not included.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Upload a PDF to start. Your file is processed in this app session.")
    else:
        pdf_bytes = uploaded_file.getvalue()
        current_digest = hashlib.sha256(
            uploaded_file.name.encode("utf-8") + b"\0" + pdf_bytes
        ).hexdigest()

        if current_digest != st.session_state.pdf_digest:
            st.session_state.pdf_digest = current_digest
            st.session_state.pdf_pages = None
            st.session_state.pdf_page_count = 0
            st.session_state.pdf_chunks = []
            st.session_state.faiss_index = None
            st.session_state.embedding_vectors = None
            st.session_state.chat_history = []

        with st.container(border=True):
            st.markdown("**Selected document**")
            st.write(uploaded_file.name)
            st.caption(f"{uploaded_file.size / 1024:.1f} KB")

            pages = st.session_state.pdf_pages
            if pages is None:
                st.markdown('<span class="status-pill">Waiting for extraction</span>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<span class="status-pill ready">{st.session_state.pdf_page_count} pages · text extracted</span>',
                    unsafe_allow_html=True,
                )
                if st.session_state.faiss_index is not None:
                    st.markdown(
                        f'<span class="status-pill ready">Index ready · {len(st.session_state.pdf_chunks)} chunks</span>',
                        unsafe_allow_html=True,
                    )
                elif any(pages):
                    st.markdown('<span class="status-pill processing">Search index not created</span>', unsafe_allow_html=True)

        if st.button("Extract PDF text", type="primary", use_container_width=True):
            st.session_state.pdf_pages = None
            st.session_state.pdf_page_count = 0
            st.session_state.pdf_chunks = []
            st.session_state.faiss_index = None
            st.session_state.embedding_vectors = None
            st.session_state.chat_history = []

            if not pdf_bytes:
                st.error("This PDF is empty (0 bytes). Choose a PDF that contains pages.")
            else:
                try:
                    with st.spinner("Reading the PDF pages…"):
                        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                            if document.needs_pass:
                                st.error("This PDF is password-protected. Choose an unprotected PDF and try again.")
                            elif document.page_count == 0:
                                st.error("This PDF has no pages. Choose a PDF with at least one page.")
                            else:
                                st.session_state.pdf_pages = [
                                    page.get_text("text").strip() for page in document
                                ]
                                st.session_state.pdf_page_count = document.page_count
                except fitz.FileDataError:
                    st.error("This file is not a readable PDF, or it may be corrupted. Choose another file.")
                except RuntimeError:
                    st.error("PyMuPDF could not read this PDF. It may be corrupted or protected.")
                except Exception:
                    st.error("We could not read this PDF. Try another valid, unprotected PDF.")

        pages = st.session_state.pdf_pages
        if pages is not None:
            if not any(pages):
                st.warning("No selectable text was found. This may be a scanned or image-only PDF; OCR is not included.")
            else:
                if st.button("Create search index", use_container_width=True):
                    try:
                        with st.status("Preparing your document for search…", expanded=True) as status:
                            chunks = chunk_pdf_pages(pages, uploaded_file.name)
                            if not chunks:
                                raise ValueError("No text chunks could be made from this PDF.")
                            st.write(f"Created {len(chunks)} page-aware text chunks.")
                            st.write("Loading the local embedding model and building the FAISS index…")
                            model = load_embedding_model()
                            index, vectors = build_faiss_index(model, chunks)
                            st.session_state.pdf_chunks = chunks
                            st.session_state.faiss_index = index
                            st.session_state.embedding_vectors = vectors
                            status.update(label="Search index is ready.", state="complete")
                    except ValueError as error:
                        st.error(str(error))
                    except Exception:
                        st.error("Could not build the search index. Check dependencies and the first-run model download.")

                if st.session_state.pdf_chunks:
                    st.caption(
                        f"{st.session_state.pdf_page_count} pages · "
                        f"{len(st.session_state.pdf_chunks)} chunks · "
                        f"{len(st.session_state.embedding_vectors)} embeddings · "
                        f"{st.session_state.faiss_index.ntotal} FAISS vectors"
                    )

                with st.expander("Preview extracted text by page"):
                    for page_number, page_text in enumerate(pages, start=1):
                        st.markdown(f"**Page {page_number}**")
                        if page_text:
                            st.text(page_text)
                        else:
                            st.caption("No selectable text on this page.")

    st.divider()
    st.caption(f"Local embeddings · {MODEL_NAME.rsplit('/', 1)[-1]}")


header_left, header_right = st.columns([5, 1])
with header_left:
    st.title("PDF Genius")
    st.markdown("<p class='muted-text' style='font-size:1.05rem;margin-top:-.8rem;'>Chat with your documents</p>", unsafe_allow_html=True)
with header_right:
    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
    if st.session_state.chat_history:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.button("Clear chat", use_container_width=True, disabled=True)

st.divider()

index_ready = (
    uploaded_file is not None
    and st.session_state.faiss_index is not None
    and bool(st.session_state.pdf_chunks)
)
api_key_available = get_api_key() is not None

if not st.session_state.chat_history:
    st.markdown(
        """
        <div class="welcome-card">
          <div class="welcome-icon">✦</div>
          <div class="welcome-title">Your document, ready for a conversation</div>
          <div class="welcome-copy">Upload a PDF in the sidebar, extract its text, and build the search index. Then ask questions in plain language and explore the answers with source pages.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Try asking")
    examples = st.columns(3)
    prompts = [
        ("Find a key idea", "What is the main idea of this document?"),
        ("Locate a detail", "What are the most important details?"),
        ("Summarize a section", "Can you summarize the key points?"),
    ]
    for column, (label, prompt) in zip(examples, prompts):
        with column:
            st.markdown(
                f'<div class="example-card"><div class="example-label">{label}</div>{prompt}</div>',
                unsafe_allow_html=True,
            )

    if not index_ready:
        st.info("To begin, upload a PDF and create its search index in the sidebar.")
    elif not api_key_available:
        st.warning("The PDF is indexed, but chat needs a Gemini API key. Add `GEMINI_API_KEY` to `.env` and restart the app.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Render source references only when the retrieval backend returned them.
        sources = message.get("sources") or []
        if sources:
            source_pairs = sorted({(item.get("source"), item.get("page_number")) for item in sources})
            source_html = " ".join(
                f'<span class="page-source">{html.escape(str(name))} · page {html.escape(str(page))}</span>'
                for name, page in source_pairs
                if name is not None and page is not None
            )
            if source_html:
                st.markdown("**Sources**  " + source_html, unsafe_allow_html=True)

if index_ready and not api_key_available and st.session_state.chat_history:
    st.info("Chat is unavailable until a Gemini API key is configured in `.env`.")

if not index_ready:
    st.chat_input("Upload and index a PDF to ask a question", disabled=True)
elif not api_key_available:
    st.chat_input("Add a Gemini API key to start chatting", disabled=True)
else:
    question = st.chat_input("Ask a question about your PDF…")
    if question is not None:
        question = question.strip()
        if not question:
            st.info("Type a question before sending it.")
        else:
            previous_history = list(st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            answer = ""
            sources = []
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching your PDF and writing an answer…"):
                        model = load_embedding_model()
                        sources = retrieve_relevant_chunks(
                            question=question,
                            model=model,
                            index=st.session_state.faiss_index,
                            chunks=st.session_state.pdf_chunks,
                            top_k=4,
                        )
                        answer = generate_pdf_answer(
                            question=question,
                            retrieved_chunks=sources,
                            chat_history=previous_history,
                        )
                except (
                    GeminiAuthenticationError,
                    GeminiModelError,
                    GeminiQuotaError,
                    GeminiSetupError,
                    GeminiRequestError,
                    ValueError,
                ) as error:
                    answer = str(error)
                except Exception:
                    answer = "I could not search the document right now. Please check the PDF index and try again."

                st.markdown(answer)
                if sources:
                    source_pairs = sorted({(item.get("source"), item.get("page_number")) for item in sources})
                    labels = [f"{name} · page {page}" for name, page in source_pairs if name and page is not None]
                    if labels:
                        st.caption("Retrieved sources: " + "; ".join(labels))

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
