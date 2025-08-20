import os
from io import BytesIO
from typing import List, Tuple, Dict, Any

import streamlit as st

# Lightweight text pipeline
from pypdf import PdfReader
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------- App config ----------
st.set_page_config(page_title="SWT Fitness — AI Support", page_icon="💬", layout="centered")

DATA_DIR = "data"
PDF_PATH = os.path.join(DATA_DIR, "knowledge.pdf")
INDEX_PATH = os.path.join(DATA_DIR, "tfidf_index.joblib")
DOCS_PATH = os.path.join(DATA_DIR, "tfidf_docs.joblib")

# --------- UI Header ----------
st.markdown(
    """
    <div style="text-align:center">
      <h1 style="margin-bottom:0">SWT Fitness — AI Support</h1>
      <p style="margin-top:6px; color:#666">
        Ask about classes, schedule, childcare, pricing, and more. Answers come only from your PDF.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------- Helpers ----------
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if not text:
        return []
    chunks, start = [], 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks

def load_pdf_to_docs(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    docs: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        for ch in chunk_text(raw):
            docs.append({"page": i, "text": ch})
    return docs

def fit_tfidf(docs: List[Dict[str, Any]]):
    corpus = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        max_features=50000,
    )
    X = vectorizer.fit_transform(corpus)  # sparse matrix
    return vectorizer, X

@st.cache_resource(show_spinner=False)
def build_or_load_index() -> Tuple[TfidfVectorizer, Any, List[Dict[str, Any]]]:
    """
    Load cached TF‑IDF index if present; otherwise build from PDF if available.
    Cached by Streamlit between reruns.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        pack = joblib.load(INDEX_PATH)
        docs = joblib.load(DOCS_PATH)
        return pack["vectorizer"], pack["matrix"], docs

    if not os.path.exists(PDF_PATH):
        raise RuntimeError("No index found. Upload a PDF and click 'Build / Rebuild Knowledge Base'.")

    docs = load_pdf_to_docs(PDF_PATH)
    if not docs:
        raise RuntimeError("Could not read any text from the PDF.")
    vectorizer, X = fit_tfidf(docs)
    joblib.dump({"vectorizer": vectorizer, "matrix": X}, INDEX_PATH)
    joblib.dump(docs, DOCS_PATH)
    return vectorizer, X, docs

def reindex_from_uploaded_pdf(file_bytes: bytes):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PDF_PATH, "wb") as f:
        f.write(file_bytes)
    # Clear cache so next call rebuilds
    build_or_load_index.clear()
    # Build once to prime cache
    build_or_load_index()

def answer_question(user_text: str) -> Tuple[str, List[str]]:
    vec, X, docs = build_or_load_index()
    q = vec.transform([user_text])
    sims = cosine_similarity(q, X)[0]
    if sims.size == 0 or np.max(sims) <= 0:
        return "Sorry, I don’t have that information in the PDF.", []
    top_idx = np.argsort(-sims)[:4].tolist()
    best = docs[top_idx[0]]["text"].strip()
    if len(best) > 1200:
        best = best[:1200] + "…"

    def label(i: int) -> str:
        p = docs[i].get("page")
        return f"Page {p + 1}" if isinstance(p, int) else "Source"

    sources = [label(i) for i in top_idx]
    return best, sources

def synthesize_tts_bytes(text: str) -> bytes | None:
    try:
        from gtts import gTTS
        buf = BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        buf.seek(0)
        return buf.read()  # MP3 bytes
    except Exception:
        return None

# --------- Knowledge base controls ----------
with st.expander("📄 Load or replace knowledge base (PDF)"):
    st.write("Upload your SWT Fitness PDF. The bot will only answer from this document.")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if st.button("Build / Rebuild Knowledge Base", type="primary", use_container_width=True):
        if not uploaded:
            st.error("Please upload a PDF first.")
        else:
            with st.spinner("Indexing PDF (chunking + TF‑IDF)…"):
                reindex_from_uploaded_pdf(uploaded.read())
            st.success("Knowledge base ready ✅")

# --------- Sidebar options ----------
with st.sidebar:
    st.subheader("Options")
    show_sources = st.toggle("Show sources", value=True)
    use_tts = st.toggle("Voice reply (gTTS)", value=False, help="Generate an audio reply.")
    st.markdown("---")
    st.caption("Runs on Streamlit free tier: TF‑IDF retriever only (no paid APIs).")

# --------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hi! I’m the SWT Fitness assistant. Ask me anything from the PDF."}
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --------- Chat input / handler ----------
def handle(q: str):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("assistant"):
        with st.spinner("Searching your PDF…"):
            try:
                reply, sources = answer_question(q)
            except Exception as e:
                st.error(f"Error: {e}")
                return
            st.markdown(reply)
            if use_tts:
                audio = synthesize_tts_bytes(reply)
                if audio:
                    st.audio(audio, format="audio/mp3")
            if show_sources and sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"- {s}")
    st.session_state.messages.append({"role": "assistant", "content": reply})

user_q = st.chat_input("Ask about schedule, memberships, childcare, etc.")
if user_q:
    handle(user_q)

st.markdown("<hr/><small>Built with Streamlit • PDF search (RAG‑only) • © SWT Fitness</small>", unsafe_allow_html=True)
