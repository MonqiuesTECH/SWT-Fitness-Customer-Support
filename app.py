# app.py — SWT Fitness Customer Support (Pro UI, safe fallbacks)
# - Simple RAG (PDF TF-IDF) when KB is loaded
# - Safe fallbacks if KB missing
# - Guided membership & schedule flows
# - Calendly handoff
# - Lead capture to Google Sheets (optional)
# - "Powered by ZARI" footer

from __future__ import annotations

import io, os, json, re, time, textwrap, datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import streamlit as st
from pypdf import PdfReader
import pytz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ---------- Optional integrations ----------
# If you installed these helpers already, keep the imports; otherwise the except keeps app running.
try:
    from calendly_api import list_available_times, create_single_use_link
except Exception:
    def list_available_times(*a, **k): return []
    def create_single_use_link(*a, **k): return None

try:
    from leads import add_lead
except Exception:
    def add_lead(*a, **k): raise RuntimeError("lead sheet not configured")

# ================== CONFIG / SECRETS ==================
st.set_page_config(page_title="SWT Fitness Customer Support", page_icon="💬", layout="wide")

STUDIO_NAME     = st.secrets.get("STUDIO_NAME", "SWT Fitness")
ADMIN_PASSWORD  = st.secrets.get("ADMIN_PASSWORD", "admin")

# Calendly
CALENDLY_EVENT_TYPE = st.secrets.get("CALENDLY_EVENT_TYPE", "").strip()
CALENDLY_URL        = st.secrets.get("CALENDLY_URL", "").strip()
CALENDLY_TZ         = st.secrets.get("CALENDLY_TZ", "America/New_York")

# ----------------- Storage paths -----------------
DATA_DIR  = "/mnt/data"
VEC_PATH  = os.path.join(DATA_DIR, "kb_vectorizer.joblib")
MAT_PATH  = os.path.join(DATA_DIR, "kb_matrix.joblib")
META_PATH = os.path.join(DATA_DIR, "kb_meta.json")

# ================== SESSION DEFAULTS ==================
ss = st.session_state
ss.setdefault("is_admin", False)
ss.setdefault("kb_ready", False)
ss.setdefault("show_sources", True)
ss.setdefault("chat_history", [])
ss.setdefault("pending_intent", None)       # for multi-turn flows
ss.setdefault("membership_map", {})         # cached parsed plans

# ================== KB TYPES & HELPERS ==================
@dataclass
class KB:
    vectorizer: TfidfVectorizer
    matrix: Any
    chunks: List[str]
    sources: List[str]

def _chunk(text: str, chunk_size: int = 500, overlap: int = 120) -> List[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += max(1, chunk_size - overlap)
    return out

def _read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    pieces = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t: pieces.append(t)
    return "\n".join(pieces)

def _build_index(from_files: List[io.BytesIO], filenames: List[str]) -> KB:
    chunks, sources = [], []
    for f, name in zip(from_files, filenames):
        f.seek(0)
        txt = _read_pdf(f)
        for ch in _chunk(txt, 500, 120):
            chunks.append(ch)
            sources.append(name)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000, lowercase=True)
    matrix = vectorizer.fit_transform(chunks)

    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(matrix, MAT_PATH)
    with open(META_PATH, "w") as fh:
        json.dump({"chunks": chunks, "sources": sources}, fh)

    return KB(vectorizer, matrix, chunks, sources)

def _load_index_if_exists() -> Optional[KB]:
    try:
        if not (os.path.exists(VEC_PATH) and os.path.exists(MAT_PATH) and os.path.exists(META_PATH)):
            return None
        vectorizer = joblib.load(VEC_PATH)
        matrix = joblib.load(MAT_PATH)
        meta = json.load(open(META_PATH))
        return KB(vectorizer, matrix, meta["chunks"], meta["sources"])
    except Exception:
        return None

def _ensure_kb() -> Optional[KB]:
    kb: Optional[KB] = ss.get("kb_obj")
    if kb: 
        return kb
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

def _compose_answer(question: str, hits: List[Tuple[str, str, float]]) -> str:
    if not hits:
        return "I don't know. Would you like to speak with our manager?"
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
    answer = textwrap.shorten(answer, width=800, placeholder="…")
    return answer

def answer_question(question: str) -> str:
    kb = _ensure_kb()
    if not kb:
        return "I don't know. Would you like to speak with our manager?"
    hits = _retrieve(kb, question, k=4)
    ans = _compose_answer(question, hits)
    if ss.show_sources and hits:
        unique = []
        for _, src, _ in hits:
            if src not in unique:
                unique.append(src)
        ans += "\n\n_sources: " + ", ".join(unique[:4])
    return ans

# ================== PARSERS & FALLBACKS ==================
def _full_text(kb: KB) -> str:
    return "\n".join(kb.chunks)

# Known fallback membership data from your flyer
FALLBACK_MEMBERSHIPS: Dict[str, Dict[str, str]] = {
    "Sapphire": {"price": "59.99", "childcare": "84.99", "classes_per_week": "2"},
    "Pearl":    {"price": "79.99", "childcare": "104.99", "classes_per_week": "3"},
    "Diamond":  {"price": "95.99", "childcare": "120.99", "classes_per_week": "Unlimited"},
}

def extract_memberships(kb: KB) -> Dict[str, Dict[str, str]]:
    """
    Best-effort parse from KB text. Falls back elsewhere if nothing found.
    """
    text = _full_text(kb)
    text = re.sub(r"\s+", " ", text)
    tiers = {"Sapphire":["Sapphire","Saphire"], "Pearl":["Pearl"], "Diamond":["Diamond","Diamonds"]}
    out: Dict[str, Dict[str, str]] = {}
    for tier, keys in tiers.items():
        pat = r"(?i)(%s)[^$]{0,40}\$?\s?([0-9]+(?:\.[0-9]{2})?)" % "|".join(keys)
        m = re.search(pat, text)
        if m:
            out[tier] = {"price": m.group(2)}
    # try to catch childcare upsells
    for tier in list(out.keys()):
        pat2 = r"(?i)%s[^$]{0,80}\$?\s?([0-9]+(?:\.[0-9]{2})?)\s*(?:with|w/)\s*child\s*care" % tier
        m2 = re.search(pat2, text)
        if m2:
            out[tier]["childcare"] = m2.group(1)
    return out

# ================== INTENT & FLOWS ==================
_MEMBERSHIP_WORDS = r"(member|plan|price|cost|join|trial)"
_SCHEDULE_WORDS   = r"(class|schedule|when|time|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"

def detect_intent(msg: str) -> str:
    if re.search(_MEMBERSHIP_WORDS, msg, re.I): return "membership"
    if re.search(_SCHEDULE_WORDS, msg, re.I):   return "schedule"
    if re.search(r"(human|person|manager|call|book|meeting|talk)", msg, re.I): return "handoff"
    return "kb"

def membership_flow(user_text: str, kb: Optional[KB]):
    # Build membership map from KB if available, else use fallback
    mm: Dict[str, Dict[str, str]] = {}
    if kb:
        try:
            mm = extract_memberships(kb)
        except Exception:
            mm = {}
    if not mm:
        mm = FALLBACK_MEMBERSHIPS

    # Price range
    prices = [float(v["price"]) for v in mm.values() if v.get("price")]
    lo, hi = (min(prices), max(prices)) if prices else (None, None)

    st.chat_message("assistant").write(
        f"Our memberships range from ${lo:.2f} to ${hi:.2f}. "
        "How many days per week do you want to come (2, 3, or Unlimited)?"
        if lo is not None else
        "We have several membership tiers. How many days per week do you want to come (2, 3, or Unlimited)?"
    )
    ss.pending_intent = ("membership_followup", mm)

def membership_followup(user_text: str, mm: Dict[str, Dict[str, str]]):
    choice = user_text.strip().lower()
    tier = None
    if re.search(r"\b2\b|two|couple", choice): tier = "Sapphire" if "Sapphire" in mm else None
    elif re.search(r"\b3\b|three", choice):    tier = "Pearl" if "Pearl" in mm else None
    elif re.search(r"unlimited|any|whenever|lot", choice): tier = "Diamond" if "Diamond" in mm else None

    if not tier:
        st.chat_message("assistant").write("I’m not sure. Would you like to speak with our manager?")
        return

    plan = mm[tier]
    price = plan.get("price", "?")
    cc    = plan.get("childcare")
    more  = f" or ${cc} with childcare" if cc else ""
    st.chat_message("assistant").write(
        f"I’d recommend **{tier}** — ${price}{more}. Want me to help you book a quick call to confirm?"
    )

def schedule_flow(user_text: str, kb: Optional[KB]):
    st.chat_message("assistant").write("What day would you like to work out?")
    ss.pending_intent = ("schedule_day", kb)

def schedule_day_followup(user_text: str, kb: Optional[KB]):
    day = user_text.strip()
    st.chat_message("assistant").write("Great — what time window works best (morning, lunchtime, evening)?")
    ss.pending_intent = ("schedule_time", (kb, day))

def schedule_time_followup(user_text: str, payload):
    kb, day = payload
    timepref = user_text.strip().lower()
    # If we have a KB, try a lightweight lookup; otherwise answer safely
    if kb:
        try:
            q = f"{day} {timepref} class schedule"
            ans = answer_question(q)  # will include sources if toggle on
            # If answer_question fell back, keep it short
            if "I don't know" in ans:
                raise RuntimeError("no kb hit")
            st.chat_message("assistant").write(ans)
            return
        except Exception:
            pass
    st.chat_message("assistant").write("I’m not sure. Would you like to speak with our manager?")

# ================== CALENDLY HANDOFF ==================
def show_manager_slots():
    st.chat_message("assistant").write("Absolutely — pick a time that works best for you:")
    tz = pytz.timezone(CALENDLY_TZ)
    now = dt.datetime.now(tz)
    start, end = now, now + dt.timedelta(days=7)

    # Use Direct URL if API unavailable or not configured
    if not CALENDLY_EVENT_TYPE:
        if CALENDLY_URL:
            st.write(f'<a href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>',
                     unsafe_allow_html=True)
        else:
            st.info("Scheduler temporarily unavailable.")
        return

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
            if url:
                st.write(f'<a href="{url}" target="_blank" rel="noopener">See more dates</a>', unsafe_allow_html=True)
                return
        except Exception:
            pass
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

# ================== UI CHROME (Pro) ==================
def _inject_css():
    st.markdown("""
    <style>
      /* Carded chat look */
      section.main > div { padding-top: 12px; }
      .stChatFloatingInputContainer { border-top: 1px solid #eee; }
      .stChatMessage { padding: 10px 14px; border-radius: 14px; margin: 8px 0; }
      .stChatMessage[data-testid="stChatMessageUser"] {
          background: #ffffff; border:1px solid #e8e8e8;
          box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      }
      .stChatMessage[data-testid="stChatMessageAssistant"] {
          background: #f7f9fc; border:1px solid #e6eef9;
      }
      /* “Powered by ZARI” footer */
      .zari { text-align:center; font-size: 12.5px; color:#667085; margin-top: 8px;}
      .zari a { color:#294bff; text-decoration:none; }
      /* Hide default header padding in narrow iframes */
      [data-testid="stHeader"] { background-color: transparent; }
    </style>
    """, unsafe_allow_html=True)

# ================== SIDEBAR ==================
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

# ================== HEADER ==================
_inject_css()
st.title(f"{STUDIO_NAME} Customer Support")
st.caption("Ask about classes, schedules, childcare, pricing, promotions, and more.")

# ================== ADMIN PANEL ==================
if ss.is_admin:
    st.subheader("Admin · Load / replace knowledge base (PDF/Text)")
    uploaded = st.file_uploader(
        "Upload one or more PDFs (pricing, policies, schedule).",
        type=["pdf"], accept_multiple_files=True)
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Build/Replace knowledge base", type="primary", use_container_width=True, disabled=(not uploaded)):
            with st.spinner("Indexing documents…"):
                kb = _build_index([io.BytesIO(f.read()) for f in uploaded], [f.name for f in uploaded])
                ss.kb_obj = kb
                ss.kb_ready = True
            st.success(f"Indexed {len(kb.chunks)} chunks from {len(set(kb.sources))} file(s).")
    with c2:
        if st.button("Load existing index from disk", use_container_width=True):
            kb = _load_index_if_exists()
            if kb:
                ss.kb_obj = kb; ss.kb_ready = True
                st.success(f"Loaded {len(kb.chunks)} chunks from disk.")
            else:
                st.warning("No saved index found yet.")
    st.markdown("---")

# ================== TURN HANDLER ==================
def handle_turn(user_text: str):
    if not user_text: return
    st.chat_message("user").write(user_text)

    # Multi-turn follow-ups first
    if ss.pending_intent:
        intent, payload = ss.pending_intent
        ss.pending_intent = None
        if intent == "membership_followup": membership_followup(user_text, payload); return
        if intent == "schedule_day":        schedule_day_followup(user_text, payload); return
        if intent == "schedule_time":       schedule_time_followup(user_text, payload); return

    # New turn → detect intent
    intent = detect_intent(user_text)
    kb = _ensure_kb()  # may be None; flows handle safely

    if intent == "membership":
        membership_flow(user_text, kb); return
    if intent == "schedule":
        schedule_flow(user_text, kb); return
    if intent == "handoff":
        show_manager_slots(); return

    # KB Q&A as a last resort
    ans = answer_question(user_text)
    st.chat_message("assistant").write(ans)
    # Lightweight lead capture when intent shows buying interest
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

# ================== CHAT LOOP ==================
for role, msg in ss.chat_history:
    st.chat_message(role).write(msg)

user_msg = st.chat_input("Type your question (schedule, memberships, childcare, etc.)") or ""
if user_msg:
    ss.chat_history.append(("user", user_msg))
    handle_turn(user_msg)

# ================== FOOTER ==================
st.markdown('<div class="zari">Powered by <strong>ZARI</strong></div>', unsafe_allow_html=True)
