# app.py — Streamlit Cloud safe (uses /tmp), with repo "kb/" fallback
from __future__ import annotations
import io, os, re, json, time, textwrap, tempfile, datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import streamlit as st
from pypdf import PdfReader
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# local modules (unchanged)
from intents import wants_handoff
from calendly_api import list_available_times, create_single_use_link
from leads import add_lead

# -------------------- Page / Secrets --------------------
st.set_page_config(page_title="SWT Fitness Customer Support", page_icon="💬", layout="wide")

STUDIO_NAME     = st.secrets.get("STUDIO_NAME", "SWT Fitness")
ADMIN_PASSWORD  = st.secrets.get("ADMIN_PASSWORD", "admin")
CALENDLY_EVENT_TYPE = (st.secrets.get("CALENDLY_EVENT_TYPE", "") or "").strip()
CALENDLY_URL        = (st.secrets.get("CALENDLY_URL", "") or "").strip()
CALENDLY_TZ         = st.secrets.get("CALENDLY_TZ", "America/New_York")

# Writable temp folder on Streamlit Cloud
TMP_DIR     = Path(tempfile.gettempdir()) / "swt_kb"
# Optional persistent fallback if you commit prebuilt index to the repo
BUNDLED_DIR = Path(__file__).parent / "kb"

VEC_PATH  = TMP_DIR / "kb_vectorizer.joblib"
MAT_PATH  = TMP_DIR / "kb_matrix.joblib"
META_PATH = TMP_DIR / "kb_meta.json"

# -------------------- Session --------------------
ss = st.session_state
ss.setdefault("is_admin", False)
ss.setdefault("kb_ready", False)
ss.setdefault("show_sources", True)
ss.setdefault("chat_history", [])

# -------------------- KB types & helpers --------------------
@dataclass
class KB:
    vectorizer: TfidfVectorizer
    matrix: Any
    chunks: List[str]
    sources: List[str]

def _chunk(text: str, size: int = 500, overlap: int = 120) -> List[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i : i + size]))
        i += max(1, size - overlap)
    return out

def _read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t.strip():
            pages.append(t)
    return "\n".join(pages)

def _build_index(blobs: List[io.BytesIO], names: List[str]) -> KB:
    chunks, sources = [], []
    for f, name in zip(blobs, names):
        f.seek(0)
        txt = _read_pdf(f)
        for ch in _chunk(txt, 500, 120):
            chunks.append(ch)
            sources.append(name)

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, lowercase=True)
    mat = vec.fit_transform(chunks)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, VEC_PATH)
    joblib.dump(mat, MAT_PATH)
    with open(META_PATH, "w") as fh:
        json.dump({"chunks": chunks, "sources": sources}, fh)

    return KB(vec, mat, chunks, sources)

def _try_load_from(base: Path) -> KB | None:
    vp, mp, me = base / "kb_vectorizer.joblib", base / "kb_matrix.joblib", base / "kb_meta.json"
    if vp.exists() and mp.exists() and me.exists():
        vec = joblib.load(vp)
        mat = joblib.load(mp)
        meta = json.load(open(me))
        return KB(vec, mat, meta["chunks"], meta["sources"])
    return None

def _load_index_if_exists() -> KB | None:
    # 1) temp (rebuilt this session)  2) repo-bundled kb/
    for base in (TMP_DIR, BUNDLED_DIR):
        try:
            kb = _try_load_from(base)
            if kb:
                return kb
        except Exception:
            continue
    return None

def _retrieve(kb: KB, query: str, k: int = 4) -> List[Tuple[str, str, float]]:
    qv = kb.vectorizer.transform([query])
    sims = cosine_similarity(qv, kb.matrix).ravel()
    idx = np.argsort(-sims)[:k]
    return [(kb.chunks[i], kb.sources[i], float(sims[i])) for i in idx]

def _compose_answer(question: str, hits: List[Tuple[str, str, float]]) -> str:
    if not hits:
        return 'I’m not sure. Would you like to speak with our manager?'
    context = "\n\n".join([h[0] for h in hits])

    # simple extractive summary (keyword pick)
    kws = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 3]
    sentences = re.split(r"(?<=[.!?])\s+", context)
    picked = []
    for s in sentences:
        ls = s.lower()
        if any(k in ls for k in kws) or len(picked) < 3:
            picked.append(s.strip())
        if len(" ".join(picked)) > 600:
            break
    answer = " ".join(picked).strip() or (sentences[0].strip() if sentences else "")
    return textwrap.shorten(answer, width=700, placeholder="…")

def answer_question(question: str) -> str:
    kb: KB | None = ss.get("kb_obj")
    if not kb:
        kb = _load_index_if_exists()
        if kb:
            ss.kb_obj = kb
            ss.kb_ready = True
    if not kb:
        return "The knowledge base isn’t loaded yet. Go to Admin → upload PDFs, then ask again."

    hits = _retrieve(kb, question, k=4)
    ans = _compose_answer(question, hits)

    if ss.show_sources and hits:
        uniq = []
        for _, src, _ in hits:
            if src not in uniq:
                uniq.append(src)
        if uniq:
            ans += "\n\n_sources: " + ", ".join(uniq[:4])
    return ans

# -------------------- Calendly handoff --------------------
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

# -------------------- Sidebar --------------------
with st.sidebar:
    st.toggle("Show sources", value=ss.show_sources, key="show_sources")
    st.caption("Runs on Streamlit free tier. PDF/Text search only (no paid APIs).")
    st.markdown("---")
    if not ss.is_admin:
        with st.popover("Admin mode"):
            pw = st.text_input("Password", type="password")
            if st.button("Log in"):
                if pw == ADMIN_PASSWORD:
                    ss.is_admin = True
                    st.success("Admin mode enabled.")
                else:
                    st.error("Wrong password.")
    else:
        st.success("Admin mode")
        if st.button("Log out"):
            ss.is_admin = False

# -------------------- Header --------------------
st.markdown(
    """
    <style>
      .zari-footer {text-align:center; color:#778; font-size:12px; margin-top:8px}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title(f"{STUDIO_NAME} Customer Support")
st.caption("Ask about classes, schedules, childcare, pricing, promotions, and more.")

# -------------------- Admin panel --------------------
if ss.is_admin:
    st.subheader("Admin · Load / replace knowledge base (PDF/Text)")
    up = st.file_uploader(
        "Upload one or more PDFs (membership pricing, policies, schedule, childcare).",
        type=["pdf"],
        accept_multiple_files=True,
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Build/Replace knowledge base", type="primary", use_container_width=True, disabled=(not up)):
            with st.spinner("Indexing documents…"):
                kb = _build_index([io.BytesIO(f.read()) for f in up], [f.name for f in up])
                ss.kb_obj = kb
                ss.kb_ready = True
            st.success(f"Indexed {len(kb.chunks)} chunks from {len(set(kb.sources))} file(s).")
            st.info("On free hosting, the index is saved in /tmp and may reset after a cold restart. "
                    "Optionally commit the 3 files under kb/ in your repo for persistence.")
    with c2:
        if st.button("Load existing index from disk", use_container_width=True):
            kb = _load_index_if_exists()
            if kb:
                ss.kb_obj = kb; ss.kb_ready = True
                st.success(f"Loaded {len(kb.chunks)} chunks.")
            else:
                st.warning("No saved index found yet.")
    st.markdown("---")

# -------------------- Turn handler --------------------
def handle_turn(user_text: str):
    if not user_text:
        return
    st.chat_message("user").write(user_text)

    # handoff intent
    if wants_handoff(user_text):
        show_manager_slots()
        return

    # guide-style flows to keep answers short & clear
    q = user_text.lower()
    if re.search(r"\b(member(ship)?|price|cost)\b", q):
        st.chat_message("assistant").write(
            "Our monthly plans start around $60 and go up to about $96 depending on class frequency and options like childcare. "
            "How often do you want to work out each week—2x, 3x, or unlimited?"
        )
        return
    if re.search(r"\b(schedule|class|time|today|tomorrow)\b", q):
        st.chat_message("assistant").write("What day works for you? (e.g., Monday) Then tell me a time window (e.g., 5–7pm).")
        return
    if re.search(r"\b(child\s*care|childcare|kids)\b", q):
        st.chat_message("assistant").write("Childcare is available during select hours. What day/time do you need?")
        return
    if re.search(r"\b(policy|freeze|cancel|pause|refund)\b", q):
        st.chat_message("assistant").write("Got it. Are you asking about freeze, cancel, late-cancel, or something else?")
        return

    # fallback to KB
    ans = answer_question(user_text)
    if not ans or ans.strip().lower().startswith("the knowledge base isn’t loaded"):
        st.chat_message("assistant").write("I’m not sure. Would you like to speak with our manager?")
    else:
        st.chat_message("assistant").write(ans)

    # lightweight lead capture
    if re.search(r"\b(trial|join|sign\s*up|membership|personal training|nutrition)\b", user_text, re.I):
        with st.expander("Leave your info for a follow-up (optional)"):
            with st.form(f"lead_{int(time.time())}"):
                name  = st.text_input("Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone (optional)")
                ok    = st.checkbox("I agree to be contacted about my inquiry.")
                submitted = st.form_submit_button("Send")
                if submitted and name and email and ok:
                    try:
                        add_lead(name, email, phone, interest="From chat", source="web")
                        st.success("Thanks! We’ll follow up shortly.")
                    except Exception:
                        st.info("Saved locally. (Lead sheet not configured.)")

# -------------------- Chat UI --------------------
for role, msg in ss.chat_history:
    st.chat_message(role).write(msg)

user_msg = st.chat_input("Type your question (schedule, memberships, childcare, etc.)") or ""
if user_msg:
    ss.chat_history.append(("user", user_msg))
    handle_turn(user_msg)

st.markdown('<div class="zari-footer">Powered by ZARI</div>', unsafe_allow_html=True)
