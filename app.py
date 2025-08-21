import os
from io import BytesIO
from typing import List, Tuple, Dict, Any, Optional
import datetime as dt, pytz, streamlit as st
from intents import wants_handoff
from calendly_api import list_available_times, create_single_use_link


import streamlit as st

# Lightweight text pipeline (no paid APIs)
from pypdf import PdfReader
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def show_manager_slots():
    if not (CALENDLY_PAT and CALENDLY_EVENT_TYPE):
        st.warning("Scheduling isn’t configured yet. Ask an admin to add CALENDLY_* secrets.")
        return

    tz = pytz.timezone(CALENDLY_TZ)
    now = dt.datetime.now(tz)
    start, end = now, now + dt.timedelta(days=7)

    st.chat_message("assistant").write("Absolutely — pick a time that works best for you:")
    try:
        slots = list_available_times(CALENDLY_PAT, CALENDLY_EVENT_TYPE, start, end, CALENDLY_TZ)
    except Exception:
        st.error("Couldn’t reach the scheduler right now. Try again in a moment.")
        if st.button("Open scheduler in a new tab"):
            url = create_single_use_link(CALENDLY_PAT, CALENDLY_EVENT_TYPE)
            st.write(f'<a href="{url}" target="_blank" rel="noopener">Open scheduler</a>', unsafe_allow_html=True)
        return

    if not slots:
        st.info("No open slots in the next 7 days.")
        if st.button("See more dates"):
            url = create_single_use_link(CALENDLY_PAT, CALENDLY_EVENT_TYPE)
            st.write(f'<a href="{url}" target="_blank" rel="noopener">Open scheduler</a>', unsafe_allow_html=True)
        return

    # Group by day and render slot buttons
    from collections import defaultdict
    by_day = defaultdict(list)
    for s in slots:
        t = dt.datetime.fromisoformat(s["start_time"].replace("Z", "+00:00")).astimezone(tz)
        by_day[t.strftime("%A %b %d")].append((t, s["scheduling_url"]))

    for day, entries in sorted(by_day.items(), key=lambda kv: kv[1][0][0]):
        with st.expander(day, expanded=len(by_day)==1):
            for t, url in entries:
                label = t.strftime("%-I:%M %p")
                st.write(
                    f'<a href="{url}" target="_blank" rel="noopener" '
                    f'style="display:inline-block;margin:6px 8px;padding:8px 12px;'
                    f'border-radius:10px;border:1px solid #ddd;text-decoration:none;">'
                    f'Book {label}</a>',
                    unsafe_allow_html=True,
                )

# ──────────────────────────────────────────────────────────────────────────────
# App / constants
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SWT Fitness Customer Support", page_icon="💪", layout="centered")
CALENDLY_PAT = st.secrets.get("CALENDLY_PAT")
CALENDLY_EVENT_TYPE = st.secrets.get("CALENDLY_EVENT_TYPE")
CALENDLY_TZ = st.secrets.get("CALENDLY_TZ", "America/New_York")

DATA_DIR  = "data"
INDEX_PATH = os.path.join(DATA_DIR, "tfidf_index.joblib")
DOCS_PATH  = os.path.join(DATA_DIR, "tfidf_docs.joblib")
MAX_PDFS   = 5

# ──────────────────────────────────────────────────────────────────────────────
# Branding header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; margin-top: 10px;">
        <h1 style="margin-bottom: 6px;">SWT Fitness Customer Support</h1>
        <div style="font-size: 16px; color:#6c757d; margin-bottom: 8px;">
            Powered by <b>ZARI</b>
        </div>
        <p style="font-size:15px; color:#444;">
            Ask about classes, schedules, childcare, pricing, promotions, and more.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Admin auth with login + logout (hardcoded password)
# ──────────────────────────────────────────────────────────────────────────────
ADMIN_PASSWORD = "MoniqueIsTheBest1994"   # <<< hardcoded password

def is_admin() -> bool:
    return st.session_state.get("is_admin", False)

def login_admin():
    with st.sidebar.form("admin_login", clear_on_submit=True):
        pwd = st.text_input("Admin password", type="password")
        ok = st.form_submit_button("Unlock")
        if ok:
            if pwd == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.sidebar.success("Admin mode unlocked ✅")
            else:
                st.sidebar.error("Incorrect password.")

def logout_admin():
    st.session_state.is_admin = False
    st.sidebar.info("Logged out of Admin mode.")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: chunking, PDF loading, TF-IDF index build/load
# ──────────────────────────────────────────────────────────────────────────────
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

def load_pdf_reader(reader: PdfReader, page_offset: int = 0) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        for ch in chunk_text(raw):
            docs.append({"page": i + page_offset, "text": ch})
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
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

@st.cache_resource(show_spinner=False)
def load_index_cached() -> Tuple[TfidfVectorizer, Any, List[Dict[str, Any]]]:
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)):
        raise RuntimeError("No knowledge base is loaded. Please ask an admin to build it.")
    pack = joblib.load(INDEX_PATH)
    docs = joblib.load(DOCS_PATH)
    return pack["vectorizer"], pack["matrix"], docs

def build_index_from_inputs(pdf_files: List[BytesIO], pasted_text: str = ""):
    os.makedirs(DATA_DIR, exist_ok=True)
    docs: List[Dict[str, Any]] = []
    total = min(len(pdf_files or []), MAX_PDFS)
    page_offset = 0
    if pdf_files:
        for f in pdf_files[:total]:
            reader = PdfReader(BytesIO(f.read()))
            file_docs = load_pdf_reader(reader, page_offset=page_offset)
            docs.extend(file_docs)
            page_offset += len(reader.pages)
    pasted_text = (pasted_text or "").strip()
    if pasted_text:
        for ch in chunk_text(pasted_text):
            docs.append({"page": -1, "text": ch})
    if not docs:
        raise RuntimeError("No content provided. Upload at least one PDF or paste text.")
    vectorizer, X = fit_tfidf(docs)
    joblib.dump({"vectorizer": vectorizer, "matrix": X}, INDEX_PATH)
    joblib.dump(docs, DOCS_PATH)
    load_index_cached.clear()

def answer_question(user_text: str) -> Tuple[str, List[str]]:
    vec, X, docs = load_index_cached()
    q = vec.transform([user_text])
    sims = cosine_similarity(q, X)[0]
    if sims.size == 0 or np.max(sims) <= 0:
        return "Sorry, I don’t have that information in the knowledge base.", []
    top_idx = np.argsort(-sims)[:4].tolist()
    best = docs[top_idx[0]]["text"].strip()
    if len(best) > 1200:
        best = best[:1200] + "…"
    def label(i: int) -> str:
        p = docs[i].get("page")
        return f"Page {p + 1}" if isinstance(p, int) and p >= 0 else "Pasted content"
    sources = [label(i) for i in top_idx]
    return best, sources

def synthesize_tts_bytes(text: str) -> Optional[bytes]:
    try:
        from gtts import gTTS
        buf = BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar (options + admin login/logout)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Options")
    show_sources = st.toggle("Show sources", value=True)
    use_tts = st.toggle("Voice reply (optional)", value=False, help="Play the answer as audio.")
    st.markdown("---")
    if is_admin():
        st.success("Admin mode")
        if st.button("Log out"):
            logout_admin()
    else:
        login_admin()
    st.caption("Runs on Streamlit free tier. PDF/Text search only (no paid APIs).")

# ──────────────────────────────────────────────────────────────────────────────
# Admin-only KB management
# ──────────────────────────────────────────────────────────────────────────────
if is_admin():
    with st.expander("🛠️ Admin • Load / replace knowledge base (PDF/Text)"):
        st.write(f"Upload **up to {MAX_PDFS} PDFs** and/or paste gym info. Rebuilding overwrites the previous index.")
        pdf_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
        pasted_text = st.text_area(
            "Optional pasted content (schedule, pricing, policies, FAQs)…",
            height=180,
            placeholder="SWT Fitness\nAddress: 10076 Southern Maryland Blvd, Dunkirk, MD\nHours: ...\nClasses: ...\nPricing: ...\nChildcare: ...\nPromotions: ...",
        )
        if st.button("Build / Rebuild Knowledge Base", type="primary", use_container_width=True):
            try:
                with st.spinner("Indexing content…"):
                    build_index_from_inputs(pdf_files or [], pasted_text=pasted_text)
                st.success("Knowledge base ready ✅")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    with st.expander("About the knowledge base"):
        st.info("This customer support tool answers from an internal knowledge base maintained by admins.")

# ──────────────────────────────────────────────────────────────────────────────
# Chat state & UI
# ──────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! How can we help today?"})

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def handle(q: str):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
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

user_q = st.chat_input("Type your question (schedule, memberships, childcare, etc.)")
if user_q:
    handle(user_q)
    user_msg = st.chat_input("Type your question (schedule, memberships, childcare, etc.)")

if user_msg:
    st.chat_message("user").write(user_msg)

    # NEW: human handoff trigger
    if wants_handoff(user_msg):
        show_manager_slots()
    else:
        # your existing RAG → LLM → answer code
        answer = answer_question(user_msg)  # whatever you already use
        st.chat_message("assistant").write(answer)
        from leads import add_lead
# ...
add_lead(name, email, phone, interest="Trial", source="web")



st.markdown("<hr/><small>© SWT Fitness • Customer Support • Powered by ZARI</small>", unsafe_allow_html=True)
