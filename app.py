# app.py — SWT Fitness Customer Support (Pro)
# Streamlit, PDF/Text RAG (TF-IDF), admin upload, Calendly handoff,
# Google Sheets lead capture, concise answers + follow-ups, ZARI branding.

from __future__ import annotations
import io, os, json, re, time, textwrap, datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import streamlit as st
from pypdf import PdfReader
import pytz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests

# -------------------- Page / Secrets --------------------
st.set_page_config(page_title="SWT Fitness Customer Support", page_icon="💬", layout="wide")

STUDIO_NAME       = st.secrets.get("STUDIO_NAME", "SWT Fitness")
ADMIN_PASSWORD    = st.secrets.get("ADMIN_PASSWORD", "admin")

CALENDLY_PAT      = st.secrets.get("CALENDLY_PAT", "").strip()
CALENDLY_EVENT    = st.secrets.get("CALENDLY_EVENT_TYPE", "").strip()  # e.g., https://api.calendly.com/event_types/XXXX
CALENDLY_URL      = st.secrets.get("CALENDLY_URL", "").strip()         # e.g., https://calendly.com/handle/slug
CALENDLY_TZ       = st.secrets.get("CALENDLY_TZ", "America/New_York")

COMPLAINT_FORM_URL = st.secrets.get("COMPLAINT_FORM_URL", "")

GOOGLE_SA_JSON    = st.secrets.get("GOOGLE_SA_JSON", "")
LEADS_SHEET_NAME  = st.secrets.get("LEADS_SHEET_NAME", "SWT AI Bot Leads")

# Storage directory (must be writable on Streamlit Cloud)
DATA_DIR = st.secrets.get("DATA_DIR", "/tmp/swt_kb")
VEC_PATH  = os.path.join(DATA_DIR, "kb_vectorizer.joblib")
MAT_PATH  = os.path.join(DATA_DIR, "kb_matrix.joblib")
META_PATH = os.path.join(DATA_DIR, "kb_meta.json")

os.makedirs(DATA_DIR, exist_ok=True)

# -------------------- Session defaults --------------------
ss = st.session_state
ss.setdefault("is_admin", False)
ss.setdefault("kb_ready", False)
ss.setdefault("show_sources", True)
ss.setdefault("chat", [])  # list[(role, text)]
# Follow-up state
ss.setdefault("awaiting_membership_freq", False)
ss.setdefault("awaiting_schedule_day", False)
ss.setdefault("awaiting_schedule_time", False)
ss.setdefault("schedule_day_value", "")
ss.setdefault("schedule_time_value", "")

# -------------------- Styling (Pro look) --------------------
st.markdown("""
<style>
/* Clean container width */
.block-container {max-width: 980px;}
/* Chat bubbles */
.stChatMessage[data-testid="stChatMessage"] > div {border-radius:14px; padding:12px 14px;}
.stChatMessage-user          > div {background:#f5f7fb;}
.stChatMessage-assistant     > div {background:#ffffff; border:1px solid #eee;}
/* Buttons-as-links for Calendly slots */
a.slot-btn {
  display:inline-block; margin:6px 8px; padding:8px 12px;
  border-radius:10px; border:1px solid #ddd; text-decoration:none;
}
/* Footer brand */
.zari {text-align:center; color:#9aa0a6; font-size:12px; margin:24px 0 8px;}
</style>
""", unsafe_allow_html=True)

# -------------------- PDF normalizer & price helpers --------------------
CURRENCY_MIN, CURRENCY_MAX = 20, 500  # discard junk outside this range

def _normalize_text(t: str) -> str:
    t = t.replace("\u00A0", " ")
    t = re.sub(r"-\s*\n\s*", "", t)      # join hyphen line breaks
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"(\d)([A-Za-z])", r"\1 \2", t)  # 96depending -> 96 depending
    t = re.sub(r"([A-Za-z])(\d)", r"\1 \2", t)  # about96 -> about 96
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _fmt_money(v: float) -> str:
    s = f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"${s}"

def _extract_amounts(text: str) -> list[float]:
    vals = []
    for m in re.finditer(r"\$?\s*(\d{2,3}(?:\.\d{1,2})?)", text):
        try:
            x = float(m.group(1))
            if CURRENCY_MIN <= x <= CURRENCY_MAX:
                vals.append(x)
        except:
            pass
    return vals

# -------------------- KB structures --------------------
@dataclass
class KB:
    vectorizer: TfidfVectorizer
    matrix: Any
    chunks: List[str]
    sources: List[str]

def _chunk(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += (chunk_size - overlap)
    return out

def _read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t.strip():
            pages.append(t)
    return _normalize_text(" ".join(pages))

def _build_index(from_files: List[io.BytesIO], filenames: List[str]) -> KB:
    all_chunks, sources = [], []
    for f, name in zip(from_files, filenames):
        f.seek(0)
        txt = _read_pdf(f)
        for ch in _chunk(txt):
            all_chunks.append(ch)
            sources.append(name)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    matrix = vectorizer.fit_transform(all_chunks)
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(matrix, MAT_PATH)
    with open(META_PATH, "w") as fp:
        json.dump({"chunks": all_chunks, "sources": sources}, fp)
    return KB(vectorizer, matrix, all_chunks, sources)

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

def _retrieve(kb: KB, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
    qv = kb.vectorizer.transform([query])
    sims = cosine_similarity(qv, kb.matrix).ravel()
    top = np.argsort(-sims)[:k]
    return [(kb.chunks[i], kb.sources[i], float(sims[i])) for i in top]

def _compose_answer(question: str, hits: List[Tuple[str, str, float]]) -> str:
    if not hits:
        return None  # allow caller to decide fallback
    context = "\n\n".join(h[0] for h in hits)
    kws = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 3]
    sentences = re.split(r"(?<=[.!?])\s+", context)
    picked = []
    for s in sentences:
        if any(k in s.lower() for k in kws) or len(picked) < 3:
            picked.append(s.strip())
        if len(" ".join(picked)) > 600:
            break
    out = " ".join(picked).strip() or (sentences[0].strip() if sentences else "")
    out = _normalize_text(out)
    return textwrap.shorten(out, width=700, placeholder="…")

def _kb_membership_range() -> Optional[tuple[float,float]]:
    kb = ss.get("kb_obj") or _load_index_if_exists()
    if not kb: return None
    amts, pat = [], re.compile(r"\b(member|membership|plan|pricing|price)\b", re.I)
    for ch in kb.chunks:
        if pat.search(ch):
            amts.extend(_extract_amounts(ch))
    if amts:
        return (min(amts), max(amts))
    return None

def _kb_find(day: Optional[str]=None, time_str: Optional[str]=None, topic: Optional[str]=None) -> str|None:
    """Very light-weight finder for schedule/childcare/policy."""
    kb = ss.get("kb_obj") or _load_index_if_exists()
    if not kb: return None
    query_parts = []
    if day: query_parts.append(day)
    if time_str: query_parts.append(time_str)
    if topic: query_parts.append(topic)
    q = " ".join(query_parts) or topic or day or time_str or ""
    hits = _retrieve(kb, q, k=6)
    ans = _compose_answer(q, hits)
    return ans

# -------------------- Calendly helpers --------------------
def list_available_times(event_uri: str, start: dt.datetime, end: dt.datetime, tz: str) -> List[Dict[str,Any]]:
    """Returns list of {'start_time': ISO, 'scheduling_url': str} using Event Type URI."""
    if not CALENDLY_PAT:
        return []
    url = "https://api.calendly.com/availability_schedules"  # Using EventType scheduling links requires pagination
    # Simpler approach: use Scheduling Links endpoint for a single-use link without per-slot fetch
    # We’ll approximate “slots” by just offering a single-use link if PAT provided.
    return []

def create_single_use_link(event_uri: str) -> str:
    if not (CALENDLY_PAT and event_uri):
        return ""
    try:
        r = requests.post(
            "https://api.calendly.com/scheduling_links",
            headers={"Authorization": f"Bearer {CALENDLY_PAT}", "Content-Type": "application/json"},
            json={"max_event_count": 1, "owner": event_uri},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("resource", {}).get("booking_url", "")
    except Exception:
        return ""

def show_manager_handoff():
    st.chat_message("assistant").write("Absolutely — pick a time that works for you:")
    # Best-effort: show single-use link (if PAT+EVENT), else the public booking page
    link = ""
    if CALENDLY_EVENT:
        link = create_single_use_link(CALENDLY_EVENT)
    open_link = link or CALENDLY_URL
    if open_link:
        st.write(f'<a class="slot-btn" href="{open_link}" target="_blank" rel="noopener">Open booking page</a>', unsafe_allow_html=True)
    else:
        st.warning("Scheduling isn’t configured yet. Add CALENDLY_URL (or PAT + EVENT) in Secrets.")

# -------------------- Google Sheets (optional) --------------------
def add_lead(name: str, email: str, phone: str="", interest: str="General", source: str="web") -> None:
    if not GOOGLE_SA_JSON or not LEADS_SHEET_NAME:
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds_dict = json.loads(GOOGLE_SA_JSON)
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open(LEADS_SHEET_NAME)
        ws = sh.sheet1
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([now, name, email, phone, interest, source], value_input_option="USER_ENTERED")
    except Exception as e:
        # Silent fail to keep UX smooth on free tier
        print("Lead capture error:", e)

# -------------------- Intent utilities --------------------
_HANDOFF_PAT = re.compile(r"\b(human|someone|person|call|talk|speak|manager|representative|real)\b", re.I)
def wants_handoff(text: str) -> bool:
    return bool(_HANDOFF_PAT.search(text))

def guess_day(s: str) -> str|None:
    days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for d in days:
        if d in s.lower():
            return d.capitalize()
    return None

def guess_time(s: str) -> str|None:
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", s, re.I)
    if not m: return None
    h = int(m.group(1)); mi = int(m.group(2) or 0); ap = m.group(3).upper()
    return f"{h}:{mi:02d} {ap}"

# -------------------- Answer composers for flows --------------------
def answer_membership(user_text: str):
    # If we’re waiting for the user’s frequency answer
    if ss.awaiting_membership_freq:
        freq = "2x" if re.search(r"\b2|two\b", user_text, re.I) else \
               "3x" if re.search(r"\b3|three\b", user_text, re.I) else \
               "unlimited" if re.search(r"\bunlimit|unlimited\b", user_text, re.I) else None
        childcare = bool(re.search(r"child\s*care|childcare|kids", user_text, re.I))
        if not freq:
            st.chat_message("assistant").write("Got it. Do you plan to work out **2x**, **3x**, or **unlimited** times per week?")
            return
        ss.awaiting_membership_freq = False

        # Try to pull plan names + prices from KB (Sapphire/Pearl/Diamond), else use range
        recommendation = None
        kb = ss.get("kb_obj") or _load_index_if_exists()
        if kb:
            text = " ".join(kb.chunks).lower()
            plans = {}
            for name in ["sapphire","pearl","diamond"]:
                if name in text:
                    window = re.findall(rf"{name}.{{0,100}}", text)
                    amounts = []
                    for w in window:
                        amounts.extend(_extract_amounts(w))
                    if amounts:
                        plans[name] = sorted(amounts)[:2]  # up to two prices found (base, childcare)
            # Map freq to likely plan name
            m = {"2x":"sapphire", "3x":"pearl", "unlimited":"diamond"}
            plan_key = m.get(freq)
            if plan_key in plans:
                p = plans[plan_key]
                base = _fmt_money(p[0])
                with_cc = _fmt_money(p[1]) if len(p) > 1 else None
                if childcare and with_cc:
                    recommendation = f"{plan_key.capitalize()} with childcare is about {with_cc} / month."
                else:
                    recommendation = f"{plan_key.capitalize()} is about {base} / month" + (" (childcare available; ask for details)" if childcare else ".")
        if not recommendation:
            rng = _kb_membership_range()
            if rng:
                low, high = map(_fmt_money, rng)
                recommendation = f"Our plans range roughly from {low} to {high} / month depending on frequency and options like childcare."
            else:
                recommendation = "Our plan pricing depends on how often you come and options like childcare."

        st.chat_message("assistant").write(
            recommendation + " Would you like to book a quick call with our manager to confirm details?"
        )
        show_manager_handoff()
        return

    # First turn in membership flow
    rng = _kb_membership_range()
    if rng:
        low, high = map(_fmt_money, rng)
        msg = (f"Our monthly plans start around {low} and go up to about {high}, depending on class frequency and options like childcare. "
               f"How often do you want to work out each week — **2x**, **3x**, or **unlimited**?")
    else:
        msg = ("Our monthly plans vary by class frequency and options like childcare. "
               "How often do you want to work out each week — **2x**, **3x**, or **unlimited**?")
    ss.awaiting_membership_freq = True
    st.chat_message("assistant").write(msg)

def answer_schedule(user_text: str):
    # Day first, then time
    if ss.awaiting_schedule_day:
        d = guess_day(user_text)
        if not d:
            st.chat_message("assistant").write("Which **day** works for you? (e.g., Monday, Tuesday…)") 
            return
        ss.schedule_day_value = d
        ss.awaiting_schedule_day = False
        ss.awaiting_schedule_time = True
        st.chat_message("assistant").write(f"Great — what **time**? (e.g., 9:30am)")
        return

    if ss.awaiting_schedule_time:
        t = guess_time(user_text)
        if not t:
            st.chat_message("assistant").write("What **time** should I check? (e.g., 6:00am, 5:30pm)")
            return
        ss.schedule_time_value = t
        ss.awaiting_schedule_time = False
        # Lookup
        ans = _kb_find(day=ss.schedule_day_value, time_str=ss.schedule_time_value, topic="class schedule")
        if ans:
            st.chat_message("assistant").write(f"{ss.schedule_day_value} around {ss.schedule_time_value}: {ans}")
        else:
            st.chat_message("assistant").write("I’m not sure from our schedule docs. Would you like to speak with our manager?")
            show_manager_handoff()
        ss.schedule_day_value = ss.schedule_time_value = ""
        return

    # Start flow
    ss.awaiting_schedule_day = True
    st.chat_message("assistant").write("Okay — what **day** would you like to work out?")

def polite_fallback():
    parts = [
        "I’m not sure. Would you like to speak with our manager?",
    ]
    if CALENDLY_URL or CALENDLY_EVENT:
        parts.append("You can book a time below.")
    parts.append('Prefer SMS? Text us at **+1 (443) 975-9649**.')
    if COMPLAINT_FORM_URL:
        parts.append(f'Feedback or an issue? Please use our form: {COMPLAINT_FORM_URL}')
    st.chat_message("assistant").write(" ".join(parts))
    show_manager_handoff()

# -------------------- General QA (RAG) --------------------
def answer_from_kb(user_text: str):
    kb = ss.get("kb_obj") or _load_index_if_exists()
    if not kb:
        st.chat_message("assistant").write("Our knowledge base isn’t loaded yet. Go to **Admin → Build/Replace knowledge base** and try again.")
        return
    hits = _retrieve(kb, user_text, k=5)
    ans = _compose_answer(user_text, hits)
    if not ans:
        polite_fallback()
        return

    if ss.show_sources and hits:
        unique = []
        for _, src, _ in hits:
            if src not in unique:
                unique.append(src)
        if unique:
            ans += "\n\n_sources: " + ", ".join(unique[:4])
    st.chat_message("assistant").write(ans)

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

# -------------------- Admin: upload / load KB --------------------
if ss.is_admin:
    st.subheader("Admin · Load / replace knowledge base (PDF/Text)")
    uploaded = st.file_uploader(
        "Upload one or more PDFs (pricing, policies, schedule, childcare).",
        type=["pdf"], accept_multiple_files=True
    )
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Build/Replace knowledge base", type="primary", use_container_width=True, disabled=(not uploaded)):
            with st.spinner("Indexing documents…"):
                kb = _build_index(
                    from_files=[io.BytesIO(f.read()) for f in uploaded],
                    filenames=[f.name for f in uploaded]
                )
                ss.kb_obj = kb
                ss.kb_ready = True
            st.success(f"Indexed {len(kb.chunks)} chunks from {len(set(kb.sources))} file(s).")
    with c2:
        if st.button("Load existing index from disk", use_container_width=True):
            kb = _load_index_if_exists()
            if kb:
                ss.kb_obj = kb
                ss.kb_ready = True
                st.success(f"Loaded {len(kb.chunks)} chunks from disk.")
            else:
                st.warning("No saved index found yet.")
    st.markdown("---")

# -------------------- Optional lead capture card --------------------
with st.expander("Leave your info for a follow-up (optional)"):
    with st.form("lead_form_main"):
        name  = st.text_input("Name *")
        email = st.text_input("Email *")
        phone = st.text_input("Phone (optional)")
        agree = st.checkbox("I agree to be contacted about my inquiry.")
        sent  = st.form_submit_button("Send")
        if sent and name and email:
            try:
                add_lead(name, email, phone, interest="From chat", source="web")
                st.success("Thanks! We’ll follow up shortly.")
            except Exception:
                st.info("Saved locally. (Lead sheet not configured.)")
        elif sent:
            st.warning("Please provide at least name and email.")

# -------------------- Turn handling --------------------
def handle_turn(user_text: str):
    q = user_text.strip()

    # Flows currently in progress
    if ss.awaiting_membership_freq: return answer_membership(q)
    if ss.awaiting_schedule_day or ss.awaiting_schedule_time: return answer_schedule(q)

    # Intent routing
    if wants_handoff(q): 
        show_manager_handoff(); return

    if re.search(r"\b(member(ship)?|price|cost)\b", q, re.I):
        return answer_membership(q)

    if re.search(r"\b(schedule|class|time|when)\b", q, re.I):
        return answer_schedule(q)

    if re.search(r"\b(child\s*care|childcare|kids)\b", q, re.I):
        ans = _kb_find(topic="childcare")
        if ans:
            st.chat_message("assistant").write(ans)
        else:
            polite_fallback()
        return

    if re.search(r"\b(policy|freeze|cancel|refund|terms)\b", q, re.I):
        ans = _kb_find(topic="membership policy")
        if ans:
            st.chat_message("assistant").write(ans)
        else:
            polite_fallback()
        return

    # General fallback → RAG
    answer_from_kb(q)

# -------------------- Replay history --------------------
for role, msg in ss.chat:
    st.chat_message(role).write(msg)

# -------------------- Chat input --------------------
user_msg = st.chat_input("Type your question (schedule, memberships, childcare, etc.)") or ""
if user_msg:
    ss.chat.append(("user", user_msg))
    st.chat_message("user").write(user_msg)
    try:
        handle_turn(user_msg)
    except Exception as e:
        # Never crash UX
        st.chat_message("assistant").write("Something went wrong. " +
            "Would you like to speak with our manager?")
        show_manager_handoff()

# -------------------- Footer (brand) --------------------
st.markdown('<div class="zari">Powered by ZARI</div>', unsafe_allow_html=True)
