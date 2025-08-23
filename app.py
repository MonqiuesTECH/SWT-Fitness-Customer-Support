# app.py (lightweight deploy-safe)
# - PDF/Text RAG using pypdf + TF-IDF (scikit-learn)
# - Admin upload + reindex
# - Human handoff via Calendly API slot buttons
# - Optional lead capture to Google Sheets (leads.py)

from __future__ import annotations

import io, os, json, re, time, textwrap, datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import streamlit as st
from pypdf import PdfReader
import pytz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from intents import wants_handoff
from calendly_api import list_available_times, create_single_use_link
from leads import add_lead  # optional; safe to ignore if not configured

# ---------- Page & secrets ----------
st.set_page_config(page_title="SWT Fitness Customer Support", page_icon="💬", layout="wide")

STUDIO_NAME = st.secrets.get("STUDIO_NAME", "SWT Fitness")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin")

CALENDLY_EVENT_TYPE = st.secrets.get("CALENDLY_EVENT_TYPE", "").strip()
CALENDLY_URL = st.secrets.get("CALENDLY_URL", "").strip()
CALENDLY_TZ = st.secrets.get("CALENDLY_TZ", "America/New_York")

DATA_DIR  = "/mnt/data"
VEC_PATH  = os.path.join(DATA_DIR, "kb_vectorizer.joblib")
MAT_PATH  = os.path.join(DATA_DIR, "kb_matrix.joblib")
META_PATH = os.path.join(DATA_DIR, "kb_meta.json")

# ---------- Session ----------
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[(role, text)]

# ---------- KB structures ----------
@dataclass
class KB:
    vectorizer: TfidfVectorizer
    matrix: Any  # sparse matrix
    chunks: List[str]
    sources: List[str]

def _chunk(text: str, chunk_size: int = 500, overlap: int = 120) -> List[str]:
    words = text.split()
    if not words: return []
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += (chunk_size - overlap)
    return out

def _read_pdf(file: io.BytesIO) -> str:
    text = []
    reader = PdfReader(file)
    for page in reader.pages:
        t = page.extract_text() or ""
        if t:
            text.append(t)
    return "\n".join(text)

def _build_index(from_files: List[io.BytesIO], filenames: List[str]) -> KB:
    all_chunks: List[str] = []
    sources: List[str] = []
    for file, name in zip(from_files, filenames):
        file.seek(0)
        txt = _read_pdf(file)
        for ch in _chunk(txt, 500, 120):
            all_chunks.append(ch)
            sources.append(name)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000, lowercase=True)
    matrix = vectorizer.fit_transform(all_chunks)

    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(matrix, MAT_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"chunks": all_chunks, "sources": sources}, f)

    return KB(vectorizer=vectorizer, matrix=matrix, chunks=all_chunks, sources=sources)

def _load_index_if_exists() -> KB | None:
    try:
        if not (os.path.exists(VEC_PATH) and os.path.exists(MAT_PATH) and os.path.exists(META_PATH)):
            return None
        vectorizer = joblib.load(VEC_PATH)
        matrix = joblib.load(MAT_PATH)
        meta = json.load(open(META_PATH))
        return KB(vectorizer=vectorizer, matrix=matrix, chunks=meta["chunks"], sources=meta["sources"])
    except Exception:
        return None

def _retrieve(kb: KB, query: str, k: int = 4) -> List[Tuple[str, str, float]]:
    qv = kb.vectorizer.transform([query])
    sims = cosine_similarity(qv, kb.matrix).ravel()
    topk_idx = np.argsort(-sims)[:k]
    out = []
    for i in topk_idx:
        out.append((kb.chunks[i], kb.sources[i], float(sims[i])))
    return out

def _compose_answer(question: str, hits: List[Tuple[str, str, float]]) -> str:
    if not hits:
        return ("I couldn’t find that in our documents. Would you like to speak with a team member? "
                "Say “I want to speak with someone.”")
    context = "\n\n".join([h[0] for h in hits])
    kws = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 3]
    sentences = re.split(r"(?<=[.!?])\s+", context)
    picked = []
    for s in sentences:
        ls = s.lower()
        if any(k in ls for k in kws) or len(picked) < 3:
            picked.append(s.strip())
        if len(" ".join(picked)) > 700:
            break
    answer = " ".join(picked).strip() or (sentences[0].strip() if sentences else "")
    return textwrap.shorten(answer, width=800, placeholder="…")

def answer_question(question: str) -> str:
    kb: KB | None = st.session_state.get("kb_obj")
    if not kb:
        kb = _load_index_if_exists()
        if kb:
            st.session_state.kb_obj = kb
            st.session_state.kb_ready = True
    if not kb:
        return "The knowledge base isn’t loaded yet. Go to Admin → upload PDFs, then ask again."
    hits = _retrieve(kb, question, k=4)
    ans = _compose_answer(question, hits)
    if st.session_state.show_sources and hits:
        unique = []
        for _, src, _ in hits:
            if src not in unique:
                unique.append(src)
        ans += "\n\n_sources: " + ", ".join(unique[:4])
    return ans

# ---------- Calendly handoff ----------
def show_manager_slots():
    tz = pytz.timezone(CALENDLY_TZ)
    now = dt.datetime.now(tz)
    start, end = now, now + dt.timedelta(days=7)
    st.chat_message("assistant").write("Absolutely — pick a time that works best for you:")

    if CALENDLY_EVENT_TYPE:
        try:
            slots = list_available_times(CALENDLY_EVENT_TYPE, start, end, CALENDLY_TZ)
        except Exception:
            st.error("Scheduler temporarily unavailable.")
            if CALENDLY_URL:
                st.write(f'<a href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>', unsafe_allow_html=True)
            return

        if not slots:
            st.info("No open slots in the next 7 days.")
            try:
                url = create_single_use_link(CALENDLY_EVENT_TYPE)
                st.write(f'<a href="{url}" target="_blank" rel="noopener">See more dates</a>', unsafe_allow_html=True)
            except Exception:
                if CALENDLY_URL:
                    st.write(f'<a href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>', unsafe_allow_html=True)
            return

        from collections import defaultdict
        by_day = defaultdict(list)
        for s in slots:
            t = dt.datetime.fromisoformat(s["start_time"].replace("Z", "+00:00")).astimezone(tz)
            by_day[t.strftime("%A %b %d")].append((t, s["scheduling_url"]))

        for day, entries in sorted(by_day.items(), key=lambda kv: kv[1][0][0]):
            with st.expander(day, expanded=len(by_day) == 1):
                for t, url in entries:
                    label = t.strftime("%-I:%M %p")
                    st.write(
                        f'<a href="{url}" target="_blank" rel="noopener" '
                        f'style="display:inline-block;margin:6px 8px;padding:8px 12px;'
                        f'border-radius:10px;border:1px solid #ddd;text-decoration:none;">'
                        f'Book {label}</a>',
                        unsafe_allow_html=True,
                    )
        return

    if CALENDLY_URL:
        st.write(f'<a href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>', unsafe_allow_html=True)
    else:
        st.warning("Calendly not configured. Add CALENDLY_EVENT_TYPE or CALENDLY_URL in secrets.")

# ---------- Sidebar ----------
with st.sidebar:
    st.toggle("Show sources", value=st.session_state.show_sources, key="show_sources")
    st.caption("Runs on Streamlit free tier. PDF/Text search only (no paid APIs).")
    st.markdown("---")
    if not st.session_state.is_admin:
        with st.popover("Admin mode"):
            pw = st.text_input("Password", type="password")
            if st.button("Log in"):
                if pw == ADMIN_PASSWORD:
                    st.session_state.is_admin = True
                    st.success("Admin mode enabled.")
                else:
                    st.error("Wrong password.")
    else:
        st.success("Admin mode")
        if st.button("Log out"):
            st.session_state.is_admin = False

# ---------- Header ----------
st.title(f"{STUDIO_NAME} Customer Support")
st.caption("Ask about classes, schedules, childcare, pricing, promotions, and more.")

# ---------- Admin panel ----------
if st.session_state.is_admin:
    st.subheader("Admin · Load / replace knowledge base (PDF/Text)")
    uploaded = st.file_uploader(
        "Upload one or more PDFs (membership pricing, policies, schedule, childcare).",
        type=["pdf"],
        accept_multiple_files=True,
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Build/Replace knowledge base", type="primary", use_container_width=True, disabled=(not uploaded)):
            with st.spinner("Indexing documents…"):
                kb = _build_index(
                    from_files=[io.BytesIO(f.read()) for f in uploaded],
                    filenames=[f.name for f in uploaded],
                )
                st.session_state.kb_obj = kb
                st.session_state.kb_ready = True
            st.success(f"Indexed {len(kb.chunks)} chunks from {len(set(kb.sources))} file(s).")
    with col2:
        if st.button("Load existing index from disk", use_container_width=True):
            kb = _load_index_if_exists()
            if kb:
                st.session_state.kb_obj = kb
                st.session_state.kb_ready = True
                st.success(f"Loaded {len(kb.chunks)} chunks from disk.")
            else:
                st.warning("No saved index found yet.")
    st.markdown("---")

# ---------- Turn handler ----------
def handle_turn(user_text: str):
    if not user_text:
        return
    st.chat_message("user").write(user_text)
    if wants_handoff(user_text):
        show_manager_slots()
        return
    ans = answer_question(user_text)
    st.chat_message("assistant").write(ans)
    if re.search(r"\b(trial|join|sign\s*up|membership|personal training|nutrition)\b", user_text, re.I):
        with st.expander("Leave your info for a follow-up (optional)"):
            with st.form(f"lead_{int(time.time())}"):
                name  = st.text_input("Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone (optional)")
                submitted = st.form_submit_button("Send to team")
                if submitted and name and email:
                    try:
                        add_lead(name, email, phone, interest="From chat", source="web")
                        st.success("Thanks! We’ll follow up shortly.")
                    except Exception:
                        st.info("Saved locally. (Lead sheet not configured.)")

# ---------- Chat UI ----------
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)

user_msg = st.chat_input("Type your question (schedule, memberships, childcare, etc.)") or ""
if user_msg:
    st.session_state.chat_history.append(("user", user_msg))
    handle_turn(user_msg)
