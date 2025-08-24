# app.py — SWT Fitness Customer Support (deploy-safe)
# - PDF/Text KB using pypdf + TF-IDF (scikit-learn)
# - Concise answers (no copy/paste). If unsure → "i don't know want to speak with our manager?"
# - Admin upload + reindex
# - Human handoff via Calendly (API slots or public link)
# - Lead capture gate (uses leads.py; safe no-op if not configured)

from __future__ import annotations

import io, json, re, time, textwrap, datetime as dt, tempfile
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

# Local modules
from intents import wants_handoff
from calendly_api import list_available_times, create_single_use_link
from leads import add_lead  # optional; will just raise handled exceptions if not configured


# -------------------- Page & Secrets --------------------
st.set_page_config(page_title="SWT Fitness Customer Support", page_icon="💬", layout="wide")

STUDIO_NAME    = st.secrets.get("STUDIO_NAME", "SWT Fitness")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin")

CALENDLY_EVENT_TYPE = st.secrets.get("CALENDLY_EVENT_TYPE", "").strip()  # API URI (preferred)
CALENDLY_URL        = st.secrets.get("CALENDLY_URL", "").strip()          # public booking link (fallback)
CALENDLY_TZ         = st.secrets.get("CALENDLY_TZ", "America/New_York")

# Confidence threshold for KB answers (lower = more tolerant)
MIN_SIMILARITY = float(st.secrets.get("MIN_SIMILARITY", 0.12))

# Storage (Streamlit Cloud free tier → use temp dir)
DATA_DIR = Path(st.secrets.get("DATA_DIR", Path(tempfile.gettempdir()) / "swt_kb")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

VEC_PATH  = DATA_DIR / "kb_vectorizer.joblib"
MAT_PATH  = DATA_DIR / "kb_matrix.joblib"
META_PATH = DATA_DIR / "kb_meta.json"

# -------------------- Session --------------------
ss = st.session_state
ss.setdefault("is_admin", False)
ss.setdefault("kb_ready", False)
ss.setdefault("show_sources", True)
ss.setdefault("chat_history", [])
ss.setdefault("lead_captured", False)
ss.setdefault("lead_profile", {})

# -------------------- KB Structures --------------------
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
    step = max(1, chunk_size - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += step
    return out

def _read_pdf(file: io.BytesIO) -> str:
    text = []
    reader = PdfReader(file)
    for page in reader.pages:
        t = page.extract_text() or ""
        if t: text.append(t)
    return "\n".join(text)

def _build_index(from_files: List[io.BytesIO], filenames: List[str]) -> KB:
    all_chunks, sources = [], []
    for file, name in zip(from_files, filenames):
        file.seek(0)
        txt = _read_pdf(file)
        for ch in _chunk(txt, 500, 120):
            all_chunks.append(ch)
            sources.append(name)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000, lowercase=True)
    matrix = vectorizer.fit_transform(all_chunks)

    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(matrix, MAT_PATH)
    META_PATH.write_text(json.dumps({"chunks": all_chunks, "sources": sources}))
    return KB(vectorizer=vectorizer, matrix=matrix, chunks=all_chunks, sources=sources)

def _load_index_if_exists() -> KB | None:
    try:
        if not (VEC_PATH.exists() and MAT_PATH.exists() and META_PATH.exists()):
            return None
        vectorizer = joblib.load(VEC_PATH)
        matrix = joblib.load(MAT_PATH)
        meta = json.loads(META_PATH.read_text())
        return KB(vectorizer=vectorizer, matrix=matrix, chunks=meta["chunks"], sources=meta["sources"])
    except Exception:
        return None

def _retrieve(kb: KB, query: str, k: int = 4) -> List[Tuple[str, str, float]]:
    qv = kb.vectorizer.transform([query])
    sims = cosine_similarity(qv, kb.matrix).ravel()
    topk_idx = np.argsort(-sims)[:k]
    return [(kb.chunks[i], kb.sources[i], float(sims[i])) for i in topk_idx]

# -------------------- Concise Answering --------------------
def _concise_answer(question: str, hits: List[Tuple[str, str, float]]) -> str | None:
    """Return a short, human answer (no verbatim paste). Return None if we can't craft one."""
    q = question.lower()
    text = "\n".join(h[0] for h in hits)

    # clean noise
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # helper
    def find_price(name: str):
        m = re.search(rf"{name}[^$]*\$\s*([0-9]+(?:\.[0-9]{{2}})?)", text, re.I)
        return m.group(1) if m else None

    # Memberships / price
    if re.search(r"\b(price|cost|membership|join|how much)\b", q):
        tiers = []
        for label in ["Sapphire", "Saphire", "Pearl", "Diamond"]:
            p = find_price(label)
            if p: tiers.append(f"{label}: ${p}")
        if tiers:
            return "Memberships — " + "; ".join(tiers) + "."

    # Hours
    if re.search(r"\b(hours?|open|close|staff(ed)?)\b", q):
        # extract a few time ranges like 8:30am–12pm
        ranges = re.findall(r'(\d{1,2}:\d{2}\s*(?:a|p)m)\s*[-–]\s*(\d{1,2}:\d{2}\s*(?:a|p)m)', text, re.I)
        uniq = []
        for a,b in ranges:
            s = f"{a}-{b}".replace(" ", "")
            if s not in uniq: uniq.append(s)
        if uniq:
            return f"Staffed hours: {', '.join(uniq[:4])}. Members have 24/7 access."

    # Childcare
    if re.search(r"\bchild\s*care|kids?\s*club\b", q):
        # look for a dollar amount near 'child care'
        m = re.search(r'(child\s*care|kids?\s*club)[^$]{0,80}\$\s*([0-9]+(?:\.[0-9]{2})?)', text, re.I)
        if m:
            return f"Childcare add-on is ${m.group(2)} (see front desk for details)."
        return "Childcare is available as an add-on; ask us for hours and ages."

    # Classes (generic)
    if re.search(r"\bclass|schedule|spin|barre|zumba|circuit|trainer|dance\b", q):
        return "Popular classes: Trainer Takeover, Silver Belles, Circuit Training, Barre, Spin, Dance & Burn."

    # Fallback: pick 1 short sentence that matches keywords
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kws = sorted(set(re.findall(r"[a-z]{4,}", q)))
    for s in sentences:
        s2 = re.sub(r'\s+', ' ', s).strip()
        if 20 <= len(s2) <= 160 and any(k in s2.lower() for k in kws):
            return s2

    return None

def answer_question(question: str) -> Tuple[str, bool]:
    """
    Returns (answer, confident).
    If not confident: caller should show the manager scheduling prompt.
    """
    kb: KB | None = ss.get("kb_obj")
    if not kb:
        kb = _load_index_if_exists()
        if kb:
            ss.kb_obj = kb
            ss.kb_ready = True
    if not kb:
        return "The knowledge base isn’t loaded yet. Go to Admin → upload PDFs, then ask again.", False

    hits = _retrieve(kb, question, k=4)
    if not hits:
        return "i don't know want to speak with our manager?", False

    max_sim = max(h[2] for h in hits)
    ans = _concise_answer(question, hits)

    if not ans or max_sim < MIN_SIMILARITY:
        return "i don't know want to speak with our manager?", False

    # append sources if enabled
    if ss.show_sources:
        uniq = []
        for _, src, _ in hits:
            if src not in uniq: uniq.append(src)
        if uniq:
            ans += "\n\n_sources: " + ", ".join(uniq[:4])

    # keep it short
    ans = textwrap.shorten(ans, width=300, placeholder="…")
    return ans, True

# -------------------- Calendly Handoff --------------------
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
                st.write(f'<a href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>',
                         unsafe_allow_html=True)
            return

        if not slots:
            st.info("No open slots in the next 7 days.")
            try:
                url = create_single_use_link(CALENDLY_EVENT_TYPE)
                st.write(f'<a href="{url}" target="_blank" rel="noopener">See more dates</a>',
                         unsafe_allow_html=True)
            except Exception:
                if CALENDLY_URL:
                    st.write(f'<a href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>',
                             unsafe_allow_html=True)
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
        st.write(f'<a href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>',
                 unsafe_allow_html=True)
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
st.title(f"{STUDIO_NAME} Customer Support")
st.caption("Ask about classes, schedules, childcare, pricing, promotions, and more.")

# -------------------- Admin Panel --------------------
if ss.is_admin:
    st.subheader("Admin · Load / replace knowledge base (PDF/Text)")
    uploaded = st.file_uploader(
        "Upload one or more PDFs (membership pricing, policies, schedule, childcare).",
        type=["pdf"], accept_multiple_files=True,
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Build/Replace knowledge base", type="primary",
                     use_container_width=True, disabled=(not uploaded)):
            with st.spinner("Indexing documents…"):
                kb = _build_index(
                    from_files=[io.BytesIO(f.read()) for f in uploaded],
                    filenames=[f.name for f in uploaded],
                )
                ss.kb_obj = kb
                ss.kb_ready = True
            st.success(f"Indexed {len(kb.chunks)} chunks from {len(set(kb.sources))} file(s).")
    with col2:
        if st.button("Load existing index from disk", use_container_width=True):
            kb = _load_index_if_exists()
            if kb:
                ss.kb_obj = kb
                ss.kb_ready = True
                st.success(f"Loaded {len(kb.chunks)} chunks from disk.")
            else:
                st.warning("No saved index found yet.")
    st.markdown("---")

# -------------------- Lead Gate --------------------
def lead_gate():
    if ss.lead_captured:
        return
    st.chat_message("assistant").write(
        "Can I get your **name**, **email**, and **phone** so we can follow up?"
    )
    with st.form("lead_gate_form", clear_on_submit=False, enter_to_submit=True):
        c1, c2 = st.columns(2)
        name  = c1.text_input("Name *")
        email = c2.text_input("Email *")
        phone = st.text_input("Phone (optional)")
        agree = st.checkbox("I agree to be contacted about my inquiry.")
        colA, colB = st.columns([1, 1])
        submit = colA.form_submit_button("Send")
        skip   = colB.form_submit_button("Skip for now")
        if submit:
            if not (name and email):
                st.warning("Name and email are required.")
            elif not agree:
                st.warning("Please check the consent box so we can contact you.")
            else:
                try:
                    ok = add_lead(name, email, phone, interest="Chat welcome", source="chat")
                    if ok:
                        st.success("Thanks! You’re all set. We’ll follow up if needed.")
                    else:
                        st.info("Saved locally. We’ll sync to Google Sheets when connected.")
                except Exception:
                    st.info("Saved locally. (Lead store not configured yet.)")
                ss.lead_captured = True
                ss.lead_profile = {"name": name, "email": email, "phone": phone}
        if skip and not ss.lead_captured:
            ss.lead_captured = True
            ss.lead_profile = {}
            st.info("No problem — you can still ask questions anytime.")

# -------------------- Turn Handler --------------------
def handle_turn(user_text: str):
    if not user_text:
        return
    st.chat_message("user").write(user_text)

    # 1) If they ask for a human
    if wants_handoff(user_text):
        show_manager_slots()
        return

    # 2) Try KB answer
    ans, confident = answer_question(user_text)
    if confident:
        st.chat_message("assistant").write(ans)
    else:
        # Required phrasing + show scheduling options
        st.chat_message("assistant").write("i don't know want to speak with our manager?")
        show_manager_slots()

    # 3) Lightweight lead hook on buying intent
    if re.search(r"\b(trial|join|sign\s*up|membership|personal training|nutrition)\b", user_text, re.I):
        with st.expander("Leave your info for a follow-up (optional)"):
            with st.form(f"lead_{int(time.time())}"):
                name  = st.text_input("Name", value=ss.lead_profile.get("name", ""))
                email = st.text_input("Email", value=ss.lead_profile.get("email", ""))
                phone = st.text_input("Phone (optional)", value=ss.lead_profile.get("phone", ""))
                submitted = st.form_submit_button("Send to team")
                if submitted and name and email:
                    try:
                        ok = add_lead(name, email, phone, interest="From chat", source="web")
                        if ok:
                            st.success("Thanks! We’ll follow up shortly.")
                        else:
                            st.info("Saved locally. (Lead sheet not configured.)")
                    except Exception:
                        st.info("Saved locally. (Lead store not configured.)")

# -------------------- Chat UI --------------------
for role, msg in ss.chat_history:
    st.chat_message(role).write(msg)

lead_gate()

user_msg = st.chat_input("Type your question (schedule, memberships, childcare, etc.)") or ""
if user_msg:
    ss.chat_history.append(("user", user_msg))
    handle_turn(user_msg)
