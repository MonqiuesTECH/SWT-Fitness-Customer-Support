# app.py — Pro chat UI, safe KB loading, null-safe flows, Calendly handoff,
# lead capture, and “Powered by ZARI” footer.

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

from intents import detect_intent, wants_handoff
from calendly_api import list_available_times, create_single_use_link
from leads import add_lead  # optional; safe if not configured

# ---------------------- Page / Theme ----------------------
st.set_page_config(
    page_title="SWT Fitness Customer Support",
    page_icon="💬",
    layout="wide",
)

# Inject a cleaner “widget-style” look
st.markdown("""
<style>
/* tighter body + nicer chat bubbles */
.main .block-container {padding-top: 1rem; padding-bottom: 3rem; max-width: 980px;}
.stChatMessage .stMarkdown p {margin: 0.25rem 0;}
/* CTA buttons inside expander */
a.slot-btn{
  display:inline-block;margin:6px 8px;padding:10px 14px;border-radius:12px;
  border:1px solid #e5e7eb;text-decoration:none;font-weight:600;
}
.small-note{font:600 12px/1.3 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; opacity:.7}
.zari-footer{margin-top:18px; padding-top:10px; border-top:1px dashed #e5e7eb; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ---------------------- Secrets ----------------------
STUDIO_NAME       = st.secrets.get("STUDIO_NAME", "SWT Fitness")
ADMIN_PASSWORD    = st.secrets.get("ADMIN_PASSWORD", "admin")
CALENDLY_EVENT_TYPE = st.secrets.get("CALENDLY_EVENT_TYPE", "").strip()
CALENDLY_URL        = st.secrets.get("CALENDLY_URL", "").strip()
CALENDLY_TZ         = st.secrets.get("CALENDLY_TZ", "America/New_York")

DATA_DIR  = "/mnt/data"
VEC_PATH  = os.path.join(DATA_DIR, "kb_vectorizer.joblib")
MAT_PATH  = os.path.join(DATA_DIR, "kb_matrix.joblib")
META_PATH = os.path.join(DATA_DIR, "kb_meta.json")

# ---------------------- Session ----------------------
ss = st.session_state
if "is_admin" not in ss:          ss.is_admin = False
if "kb_ready" not in ss:          ss.kb_ready = False
if "show_sources" not in ss:      ss.show_sources = True
if "chat_history" not in ss:      ss.chat_history = []  # list[(role, text)]
if "membership_map" not in ss:    ss.membership_map = {}
if "sched_step" not in ss:        ss.sched_step = None
if "sched_day" not in ss:         ss.sched_day = None

# ---------------------- KB structures ----------------------
@dataclass
class KB:
    vectorizer: TfidfVectorizer
    matrix: Any  # sparse
    chunks: List[str]
    sources: List[str]

def _chunk(text: str, chunk_size: int = 500, overlap: int = 120) -> List[str]:
    words = text.split()
    if not words: return []
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += max(1, chunk_size - overlap)
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
            all_chunks.append(ch); sources.append(name)
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

def get_kb():
    """Lazy-load the KB from disk on first use."""
    kb = ss.get("kb_obj")
    if kb: return kb
    kb = _load_index_if_exists()
    if kb:
        ss.kb_obj = kb
        ss.kb_ready = True
    return kb

def _retrieve(kb: KB, query: str, k: int = 4) -> List[Tuple[str, str, float]]:
    qv = kb.vectorizer.transform([query])
    sims = cosine_similarity(qv, kb.matrix).ravel()
    idx = np.argsort(-sims)[:k]
    return [(kb.chunks[i], kb.sources[i], float(sims[i])) for i in idx]

def _compose_answer(question: str, hits: List[Tuple[str, str, float]], min_conf: float = 0.08) -> str:
    if not hits or (hits[0][2] < min_conf):
        return ""
    context = "\n\n".join([h[0] for h in hits])
    # extract a few relevant sentences rather than dump text
    kws = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 3]
    sents = re.split(r"(?<=[.!?])\s+", context)
    picked = []
    for s in sents:
        ls = s.lower()
        if any(k in ls for k in kws) or len(picked) < 3:
            picked.append(s.strip())
        if len(" ".join(picked)) > 700: break
    return textwrap.shorten(" ".join(picked).strip(), width=800, placeholder="…")

def answer_question(question: str) -> str:
    kb = get_kb()
    if not kb:  # no index yet
        return ""
    hits = _retrieve(kb, question, k=4)
    ans = _compose_answer(question, hits)
    if not ans:
        return ""
    if ss.show_sources and hits:
        unique = []
        for _, src, _ in hits:
            if src not in unique: unique.append(src)
        ans += "\n\n" + f'<span class="small-note">sources: ' + ", ".join(unique[:4]) + "</span>"
    return ans

# ---------------------- Extraction helpers ----------------------
def extract_memberships(kb: KB) -> Dict[str, Dict[str, Any]]:
    """Very light parser for plan names + $prices."""
    if kb is None: return {}
    plans: Dict[str, Dict[str, Any]] = {}
    price_re = re.compile(r"\$?\s*(\d{2,3}(?:\.\d{2})?)")
    for ch in kb.chunks:
        lower = ch.lower()
        for name in ["sapphire","saphire","pearl","diamond","basic","standard","plus","premium","unlimited"]:
            if name in lower:
                m = price_re.search(ch)
                price = float(m.group(1)) if m else None
                title = name.capitalize()
                plans.setdefault(title, {})["price"] = price
                plans[title]["raw"] = ch
    return plans

def find_classes_for(kb: KB, day: str, when: str | None = None) -> List[str]:
    """Heuristic: pull lines mentioning the day and show time + class names."""
    if kb is None: return []
    day_rx = re.compile(day, re.I)
    time_rx = re.compile(r"\b(\d{1,2}[:.]?\d{0,2}\s?(?:am|pm))\b", re.I)
    results = []
    for ch in kb.chunks:
        if not day_rx.search(ch): continue
        # Split into lines and pick those with times
        for line in ch.splitlines():
            if day_rx.search(line) or time_rx.search(line):
                results.append(line.strip())
    # Deduplicate and clean
    out, seen = [], set()
    for r in results:
        r = re.sub(r"\s{2,}", " ", r)
        if r not in seen:
            out.append(r); seen.add(r)
    return out[:12]

# ---------------------- Calendly handoff ----------------------
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
                st.write(f'<a class="slot-btn" href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>', unsafe_allow_html=True)
            return

        if not slots:
            st.info("No open slots in the next 7 days.")
            try:
                url = create_single_use_link(CALENDLY_EVENT_TYPE)
                st.write(f'<a class="slot-btn" href="{url}" target="_blank" rel="noopener">See more dates</a>', unsafe_allow_html=True)
            except Exception:
                if CALENDLY_URL:
                    st.write(f'<a class="slot-btn" href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>', unsafe_allow_html=True)
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
                    st.write(f'<a class="slot-btn" href="{url}" target="_blank" rel="noopener">Book {label}</a>', unsafe_allow_html=True)
        return

    if CALENDLY_URL:
        st.write(f'<a class="slot-btn" href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>', unsafe_allow_html=True)
    else:
        st.warning("Calendly not configured. Add CALENDLY_EVENT_TYPE or CALENDLY_URL in secrets.")

# ---------------------- Sidebar ----------------------
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

# ---------------------- Header ----------------------
st.title(f"{STUDIO_NAME} Customer Support")
st.caption("Ask about classes, schedules, childcare, pricing, promotions, and more.")

# ---------------------- Admin: upload / load KB ----------------------
if ss.is_admin:
    st.subheader("Admin · Load / replace knowledge base (PDF/Text)")
    uploaded = st.file_uploader(
        "Upload one or more PDFs (membership pricing, policies, schedule, childcare).",
        type=["pdf"], accept_multiple_files=True,
    )
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Build/Replace knowledge base", type="primary", use_container_width=True, disabled=(not uploaded)):
            with st.spinner("Indexing documents…"):
                kb = _build_index(
                    from_files=[io.BytesIO(f.read()) for f in uploaded],
                    filenames=[f.name for f in uploaded],
                )
                ss.kb_obj = kb
                ss.kb_ready = True
                ss.membership_map = {}  # reset caches
                ss.sched_step = None; ss.sched_day = None
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

# ---------------------- Conversational Flows ----------------------
def _kb_not_ready_msg():
    st.chat_message("assistant").write("I don't know — would you like to speak with our manager?")
    show_manager_slots()

def membership_flow(user_text: str, kb: KB | None):
    if kb is None or not getattr(kb, "chunks", None):
        _kb_not_ready_msg(); return
    if not ss.membership_map:
        ss.membership_map = extract_memberships(kb)

    mm = ss.membership_map or {}
    prices = [float(v["price"]) for v in mm.values() if v.get("price") is not None]
    if prices:
        lo, hi = min(prices), max(prices)
        st.chat_message("assistant").write(
            f"Our memberships range from **${lo:,.2f}–${hi:,.2f}**. "
            "How many days per week do you want to work out?"
        )
    else:
        _kb_not_ready_msg()

def schedule_flow(user_text: str, kb: KB | None):
    if kb is None or not getattr(kb, "chunks", None):
        _kb_not_ready_msg(); return

    # step 1: ask for day
    if ss.sched_step is None:
        ss.sched_step = "ask_day"
        st.chat_message("assistant").write("Okay — what **day** would you like to work out? (e.g., Monday)")
        return

    # parse day from user
    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    if ss.sched_step == "ask_day":
        for d in day_names:
            if re.search(fr"\b{d}\b", user_text, re.I):
                ss.sched_day = d
                ss.sched_step = "show"
                break
        if ss.sched_step != "show":
            st.chat_message("assistant").write("Got it — which **day**? (e.g., Tuesday)")
            return

    # show matches
    if ss.sched_step == "show" and ss.sched_day:
        lines = find_classes_for(kb, ss.sched_day)
        if not lines:
            st.chat_message("assistant").write(
                f"I don't know what's scheduled for **{ss.sched_day}**. Would you like to speak with our manager?"
            )
            show_manager_slots()
        else:
            st.chat_message("assistant").markdown(
                f"**{ss.sched_day} — here’s what we have:**\n\n- " + "\n- ".join(lines)
            )
        # reset flow
        ss.sched_step = None
        ss.sched_day = None

# ---------------------- Turn handler ----------------------
def handle_turn(user_text: str):
    if not user_text: return
    st.chat_message("user").write(user_text)

    kb = get_kb()  # try to load on demand

    # manager handoff phrases
    if wants_handoff(user_text):
        show_manager_slots(); return

    intent = detect_intent(user_text)

    if intent == "membership":
        membership_flow(user_text, kb); return

    if intent == "schedule":
        schedule_flow(user_text, kb); return

    # general QA from KB with confidence threshold
    ans = answer_question(user_text)
    if ans:
        st.chat_message("assistant").markdown(ans, unsafe_allow_html=True)
    else:
        _kb_not_ready_msg()

    # lightweight lead capture nudge on intent-y keywords
    if re.search(r"\b(trial|join|sign\s*up|membership|personal training|nutrition)\b", user_text, re.I):
        with st.expander("Leave your info for a follow-up (optional)"):
            with st.form(f"lead_{int(time.time())}"):
                name  = st.text_input("Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone (optional)")
                optin = st.checkbox("I agree to be contacted about my inquiry.")
                submitted = st.form_submit_button("Send")
                if submitted and name and email and optin:
                    try:
                        add_lead(name, email, phone, interest="From chat", source="web")
                        st.success("Thanks! We’ll follow up shortly.")
                    except Exception:
                        st.info("Saved locally. (Lead sheet not configured.)")

# ---------------------- Replay + Input ----------------------
for role, msg in ss.chat_history:
    st.chat_message(role).write(msg)

user_msg = st.chat_input("Type your question (schedule, memberships, childcare, etc.)") or ""
if user_msg:
    ss.chat_history.append(("user", user_msg))
    handle_turn(user_msg)

# ---------------------- Footer ----------------------
st.markdown(
    '<div class="zari-footer small-note">Powered by <strong>ZARI</strong></div>',
    unsafe_allow_html=True
)
