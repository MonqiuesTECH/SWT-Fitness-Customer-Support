# app.py — SWT Fitness Customer Support (Pro UI, durable, guided flows)
# - Guided flows for memberships, schedule, childcare, policy (follow-up first)
# - Calendly handoff + SMS alternative + Feedback form
# - RAG over uploaded PDFs (TF-IDF) with safe fallbacks
# - Admin upload panel
# - "Powered by ZARI" footer
# - Cold-start safe (never touches KB if not loaded)

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

# Contact options
SWT_SMS_NUMBER     = "+1 (443) 975 - 9649"

# Storage for KB files
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
ss.setdefault("pending_intent", None)  # ("intent_name", payload)

# ================== SAFE FALLBACK DATA ==================
DEFAULT_MEMBERSHIPS = {
    "Sapphire": {"price": "59.99", "classes_per_week": "2", "childcare": "84.99"},
    "Pearl":    {"price": "79.99", "classes_per_week": "3", "childcare": "104.99"},
    "Diamond":  {"price": "95.99", "classes_per_week": "Unlimited", "childcare": "120.99"},
}

# ================== KB HELPERS ==================
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
        out.append(" ".join(words[i:i+size]))
        i += max(1, size - overlap)
    return out

def _read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    parts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t: parts.append(t)
    return "\n".join(parts)

def _build_index(files: List[io.BytesIO], names: List[str]) -> KB:
    chunks, sources = [], []
    for f, nm in zip(files, names):
        f.seek(0)
        txt = _read_pdf(f)
        for ch in _chunk(txt):
            chunks.append(ch); sources.append(nm)
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=50000, lowercase=True)
    mat = vec.fit_transform(chunks)

    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(vec, VEC_PATH)
    joblib.dump(mat, MAT_PATH)
    with open(META_PATH, "w") as fh:
        json.dump({"chunks": chunks, "sources": sources}, fh)
    return KB(vec, mat, chunks, sources)

def _load_index_if_exists() -> Optional[KB]:
    try:
        if not (os.path.exists(VEC_PATH) and os.path.exists(MAT_PATH) and os.path.exists(META_PATH)):
            return None
        vec  = joblib.load(VEC_PATH)
        mat  = joblib.load(MAT_PATH)
        meta = json.load(open(META_PATH))
        return KB(vec, mat, meta["chunks"], meta["sources"])
    except Exception:
        return None

def _get_kb() -> Optional[KB]:
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
        if len(" ".join(picked)) > 700: break
    ans = " ".join(picked).strip() or (sentences[0].strip() if sentences else "")
    return textwrap.shorten(ans, width=800, placeholder="…")

def answer_from_kb(question: str) -> str:
    kb = _get_kb()
    if not kb:
        return "I don't know. Would you like to speak with our manager?"
    hits = _retrieve(kb, question, k=4)
    ans = _compose_answer(question, hits)
    if ss.show_sources and hits:
        uniq = []
        for _, s, _ in hits:
            if s not in uniq: uniq.append(s)
        ans += "\n\n_sources: " + ", ".join(uniq[:4])
    return ans

# ================== LIGHT PARSERS ==================
def _full_text(kb: KB) -> str:
    return "\n".join(kb.chunks)

def extract_memberships(kb: Optional[KB]) -> Dict[str, Dict[str, str]]:
    if not kb: return {}
    t = re.sub(r"\s+", " ", _full_text(kb))
    out: Dict[str, Dict[str, str]] = {}
    tiers = {"Sapphire":["Sapphire","Saphire"], "Pearl":["Pearl"], "Diamond":["Diamond","Diamonds"]}
    for tier, keys in tiers.items():
        m = re.search(rf"(?i)({'|'.join(keys)})[^$]{{0,40}}\$?\s?([0-9]+(?:\.[0-9]{{2}})?)", t)
        if m: out[tier] = {"price": m.group(2)}
        m2 = re.search(rf"(?i){tier}[^$]{{0,80}}\$?\s?([0-9]+(?:\.[0-9]{{2}})?)\s*(with|w/)\s*child\s*care", t)
        if m2: out.setdefault(tier, {})["childcare"] = m2.group(1)
    return out

def extract_policy_snippet(kb: Optional[KB], topic: str) -> Optional[str]:
    if not kb: return None
    t = _full_text(kb)
    # naive window around keywords
    key = {"cancel":"cancel", "freeze":"freeze", "late":"late cancel"}.get(topic, topic)
    m = re.search(rf"(?is).{{0,240}}{re.escape(key)}.{{0,240}}", t)
    if not m: return None
    s = re.sub(r"\s+", " ", m.group(0)).strip()
    return textwrap.shorten(s, 500, placeholder="…")

# ================== INTENTS ==================
_MEMBERSHIP_WORDS = r"(member|plan|price|cost|join)"
_SCHEDULE_WORDS   = r"(class|schedule|when|time|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_CHILDCARE_WORDS  = r"(child\s*care|childcare|kids? club|babysit)"
_POLICY_WORDS     = r"(policy|cancel|freeze|late\s*cancel|terms|rules)"
_HANDOFF_WORDS    = r"(human|person|manager|call|book|meeting|talk|speak)"

def detect_intent(msg: str) -> str:
    m = msg.lower()
    if re.search(_HANDOFF_WORDS, m):   return "handoff"
    if re.search(_MEMBERSHIP_WORDS, m):return "membership"
    if re.search(_CHILDCARE_WORDS, m): return "childcare"
    if re.search(_POLICY_WORDS, m):    return "policy"
    if re.search(_SCHEDULE_WORDS, m):  return "schedule"
    return "kb"

# ================== COMMON CONTACT OPTIONS ==================
def show_contact_options():
    st.chat_message("assistant").write(
        f"If you’d prefer texting, send us a message at **{SWT_SMS_NUMBER}** and we’ll reply within ~30 minutes."
    )
    with st.expander("Complaint / Feedback (opens a short form)"):
        with st.form(f"fb_{int(time.time())}"):
            name  = st.text_input("Name")
            email = st.text_input("Email")
            phone = st.text_input("Phone (optional)")
            issue = st.text_area("Describe your complaint or feedback")
            ok    = st.checkbox("I agree to be contacted about my submission.")
            submitted = st.form_submit_button("Submit")
            if submitted and name and email and issue and ok:
                try:
                    add_lead(name, email, phone, interest=f"Feedback: {issue[:80]}", source="feedback")
                    st.success("Thanks! We received your feedback and will follow up.")
                except Exception:
                    st.info("Saved locally. (Feedback sheet not configured.)")

    st.chat_message("assistant").write("Would you like to pick a time with our manager?")
    show_manager_slots()

# ================== FLOWS ==================
def membership_flow():
    kb = _get_kb()
    mm = extract_memberships(kb) or DEFAULT_MEMBERSHIPS
    try:
        prices = [float(v["price"]) for v in mm.values() if "price" in v]
        lo, hi = min(prices), max(prices)
        pr = f"${lo:.2f}–${hi:.2f}"
    except Exception:
        pr = "multiple price points"
    st.chat_message("assistant").write(
        f"Our memberships range from {pr}. **How many days per week** do you want to come (2, 3, or Unlimited)?"
    )
    ss.pending_intent = ("membership_followup", mm)

def membership_followup(user_text: str, mm: Dict[str, Dict[str, str]]):
    choice = user_text.strip().lower()
    tier = None
    if re.search(r"\b2\b|two", choice): tier = "Sapphire" if "Sapphire" in mm else None
    elif re.search(r"\b3\b|three", choice): tier = "Pearl" if "Pearl" in mm else None
    elif re.search(r"unlimited|any|whenever", choice): tier = "Diamond" if "Diamond" in mm else None
    if not tier:
        st.chat_message("assistant").write("I’m not sure which plan that is. Would you like to speak with our manager?")
        show_contact_options()
        return
    info  = mm[tier]
    price = info.get("price", "—")
    cc    = info.get("childcare")
    cc_txt = f" (with childcare: ${cc})" if cc else ""
    st.chat_message("assistant").write(
        f"I’d recommend **{tier}** at **${price}**/mo{cc_txt}. Do you want to see class options next?"
    )
    # After a recommendation, present contact choices
    show_contact_options()

def schedule_flow():
    st.chat_message("assistant").write("**What day** would you like to work out?")
    ss.pending_intent = ("schedule_day", None)

def schedule_day_followup(day_text: str):
    day = day_text.strip()
    st.chat_message("assistant").write("Great — **what time** works best (morning, lunchtime, evening)?")
    ss.pending_intent = ("schedule_time", day)

def schedule_time_followup(time_text: str, day: str):
    pref = time_text.strip().lower()
    kb = _get_kb()
    if kb:
        try:
            q = f"{day} {pref} class schedule"
            ans = answer_from_kb(q)
            if "I don't know" not in ans:
                st.chat_message("assistant").write(ans)
            else:
                st.chat_message("assistant").write("I’m not sure. Would you like to speak with our manager?")
                show_contact_options()
                return
        except Exception:
            st.chat_message("assistant").write("I’m not sure. Would you like to speak with our manager?")
            show_contact_options()
            return
    else:
        st.chat_message("assistant").write("Our detailed schedule isn’t loaded yet. Would you like to speak with our manager?")
    show_contact_options()

def childcare_flow():
    st.chat_message("assistant").write("For childcare, **what time of day** do you prefer, and the **child’s age**?")
    ss.pending_intent = ("childcare_followup", None)

def childcare_followup(user_text: str):
    kb = _get_kb()
    if kb:
        q = f"childcare availability {user_text}"
        ans = answer_from_kb(q)
        st.chat_message("assistant").write(ans)
    else:
        st.chat_message("assistant").write(
            "Childcare details vary by time and age. I can confirm availability for you."
        )
    show_contact_options()

def policy_flow():
    st.chat_message("assistant").write(
        "Which policy do you need? **Cancel**, **Freeze**, or **Late Cancel**?"
    )
    ss.pending_intent = ("policy_followup", None)

def policy_followup(user_text: str):
    kb = _get_kb()
    topic = "cancel" if "cancel" in user_text.lower() and "late" not in user_text.lower() else \
            "freeze" if "freeze" in user_text.lower() else \
            "late"
    snippet = extract_policy_snippet(kb, topic)
    if snippet:
        st.chat_message("assistant").write(snippet)
    else:
        st.chat_message("assistant").write("I don’t know. Would you like to speak with our manager?")
    show_contact_options()

def handoff_flow():
    st.chat_message("assistant").write("Absolutely — pick a time that works best for you:")
    show_manager_slots()

# ================== CALENDLY ==================
def show_manager_slots():
    tz = pytz.timezone(CALENDLY_TZ)
    now = dt.datetime.now(tz)
    start, end = now, now + dt.timedelta(days=7)

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

# ================== UI / THEME ==================
def _inject_css():
    st.markdown("""
    <style>
      section.main > div { padding-top: 10px; }
      .stChatMessage { padding: 10px 14px; border-radius: 14px; margin: 8px 0; }
      .stChatMessage[data-testid="stChatMessageUser"] {
          background: #ffffff; border:1px solid #e8e8e8;
          box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      }
      .stChatMessage[data-testid="stChatMessageAssistant"] {
          background: #f7f9fc; border:1px solid #e6eef9;
      }
      .zari { text-align:center; font-size: 12.5px; color:#667085; margin-top: 8px;}
      .zari strong { color:#2e3aef; }
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
                    ss.is_admin = True; st.success("Admin mode enabled.")
                else:
                    st.error("Wrong password.")
    else:
        st.success("Admin mode")
        if st.button("Log out"): ss.is_admin = False

# ================== HEADER ==================
_inject_css()
st.title(f"{STUDIO_NAME} Customer Support")
st.caption("Ask about classes, schedules, childcare, pricing, policies, and more.")

# Helpful notice if KB not loaded yet
if not _get_kb():
    st.info("Heads up: Knowledge Base isn’t loaded yet. Admins can upload PDFs in the sidebar. The bot still answers basic membership questions.")

# ================== ADMIN PANEL ==================
if ss.is_admin:
    st.subheader("Admin · Load / replace knowledge base (PDF/Text)")
    up = st.file_uploader("Upload one or more PDFs (pricing, policies, schedule).", type=["pdf"], accept_multiple_files=True)
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Build/Replace knowledge base", type="primary", use_container_width=True, disabled=(not up)):
            with st.spinner("Indexing documents…"):
                kb = _build_index([io.BytesIO(f.read()) for f in up], [f.name for f in up])
                ss.kb_obj = kb; ss.kb_ready = True
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

    # Handle ongoing multi-turn step first
    if ss.pending_intent:
        intent, payload = ss.pending_intent
        ss.pending_intent = None
        if intent == "membership_followup": membership_followup(user_text, payload); return
        if intent == "schedule_day":        schedule_day_followup(user_text); return
        if intent == "schedule_time":       schedule_time_followup(user_text, payload); return
        if intent == "childcare_followup":  childcare_followup(user_text); return
        if intent == "policy_followup":     policy_followup(user_text); return

    # New turn → detect intent
    intent = detect_intent(user_text)
    if intent == "handoff":    handoff_flow(); return
    if intent == "membership": membership_flow(); return
    if intent == "schedule":   schedule_flow(); return
    if intent == "childcare":  childcare_flow(); return
    if intent == "policy":     policy_flow(); return

    # KB fallback
    ans = answer_from_kb(user_text)
    st.chat_message("assistant").write(ans)
    if "I don't know" in ans:
        show_contact_options()

# ================== CHAT LOOP ==================
for role, msg in ss.chat_history:
    st.chat_message(role).write(msg)

user_msg = st.chat_input("Type your question (schedule, memberships, childcare, policies, etc.)") or ""
if user_msg:
    ss.chat_history.append(("user", user_msg))
    handle_turn(user_msg)

# ================== FOOTER ==================
st.markdown('<div class="zari">Powered by <strong>ZARI</strong></div>', unsafe_allow_html=True)
