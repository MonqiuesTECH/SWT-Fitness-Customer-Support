# app.py — SWT Fitness Customer Support (Pro UI)
# - PDF/Text RAG (pypdf + TF-IDF) stored on /mount/data (Streamlit Cloud-safe)
# - Admin upload & reindex
# - Guided intents (membership, schedule) with follow-ups (no copy/paste walls)
# - Human handoff via Calendly (slot buttons + fallback booking link)
# - Optional lead capture to Google Sheets
# - Feedback form link (secrets) and escalation to text/manager
# - "Powered by ZARI" footer

from __future__ import annotations

import io, os, re, json, time, textwrap, datetime as dt, tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pytz

# ---------- Optional deps (safe stubs if modules missing) ----------
try:
    from intents import wants_handoff as _wants_handoff_external
except Exception:
    def _wants_handoff_external(t: str) -> bool:
        return bool(re.search(r"\b(speak|talk|call|human|person|manager|someone)\b", t, re.I))

try:
    from calendly_api import list_available_times, create_single_use_link
except Exception:
    def list_available_times(event_type_uri: str, start: dt.datetime, end: dt.datetime, tz: str) -> List[Dict]:
        return []
    def create_single_use_link(event_type_uri: str) -> str:
        return ""

try:
    from leads import add_lead
except Exception:
    def add_lead(name: str, email: str, phone: str = "", interest: str = "", source: str = ""):
        return

# ---------- Page & secrets ----------
st.set_page_config(page_title="SWT Fitness Customer Support", page_icon="💬", layout="wide")

STUDIO_NAME        = st.secrets.get("STUDIO_NAME", "SWT Fitness")
ADMIN_PASSWORD     = st.secrets.get("ADMIN_PASSWORD", "admin")
CALENDLY_EVENT_TYPE= st.secrets.get("CALENDLY_EVENT_TYPE", "").strip()
CALENDLY_URL       = st.secrets.get("CALENDLY_URL", "").strip()
CALENDLY_TZ        = st.secrets.get("CALENDLY_TZ", "America/New_York")
FEEDBACK_FORM_URL  = st.secrets.get("FEEDBACK_FORM_URL", "").strip()  # optional

# ---------- Storage paths (Streamlit Cloud-safe) ----------
def _pick_writable_dir() -> Path:
    candidates = [
        Path(os.environ.get("DATA_DIR", "/mount/data")) / "swt_kb",
        Path.cwd() / ".kb_cache",
        Path(tempfile.gettempdir()) / "swt_kb",
    ]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            p = c / ".write_test"
            p.write_text("ok")
            p.unlink(missing_ok=True)
            return c
        except Exception:
            continue
    raise RuntimeError("No writable data directory available")

DATA_DIR  = _pick_writable_dir()
VEC_PATH  = DATA_DIR / "kb_vectorizer.joblib"
MAT_PATH  = DATA_DIR / "kb_matrix.joblib"
META_PATH = DATA_DIR / "kb_meta.json"

# ---------- Session ----------
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[(role, text)]
if "flow" not in st.session_state:
    st.session_state.flow = {}          # state for guided intents

# ---------- Styles (Pro-ish look) ----------
st.markdown("""
<style>
/* cleaner chat bubbles */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { margin: 0.2rem 0; }
.stChatFloatingInputContainer { border-top: 1px solid #eee; }
.badge { display:inline-block; padding:2px 8px; border:1px solid #ddd; border-radius:999px; font-size:12px; color:#666; }
.slot-btn { display:inline-block; margin:6px 8px; padding:8px 12px; border-radius:10px; border:1px solid #ddd; text-decoration:none; }
.footer { text-align:center; color:#888; font-size:12px; padding:8px 0 0 0; }
.brand { font-weight:600; color:#0f766e; }
</style>
""", unsafe_allow_html=True)

# ---------- KB structures ----------
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
    reader = PdfReader(file)
    parts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t: parts.append(t)
    return "\n".join(parts)

def _build_index(from_files: List[io.BytesIO], filenames: List[str]) -> KB:
    all_chunks, sources = [], []
    for f, name in zip(from_files, filenames):
        f.seek(0)
        txt = _read_pdf(f)
        for ch in _chunk(txt, 500, 120):
            all_chunks.append(ch)
            sources.append(name)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000, lowercase=True)
    matrix = vectorizer.fit_transform(all_chunks)

    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(matrix, MAT_PATH)
    META_PATH.write_text(json.dumps({"chunks": all_chunks, "sources": sources}))
    return KB(vectorizer, matrix, all_chunks, sources)

def _load_index_if_exists() -> KB | None:
    try:
        if not (VEC_PATH.exists() and MAT_PATH.exists() and META_PATH.exists()):
            return None
        vectorizer = joblib.load(VEC_PATH)
        matrix = joblib.load(MAT_PATH)
        meta = json.loads(META_PATH.read_text())
        return KB(vectorizer, matrix, meta["chunks"], meta["sources"])
    except Exception:
        return None

def _retrieve(kb: KB, query: str, k: int = 4) -> List[Tuple[str, str, float]]:
    qv = kb.vectorizer.transform([query])
    sims = cosine_similarity(qv, kb.matrix).ravel()
    idx = np.argsort(-sims)[:k]
    return [(kb.chunks[i], kb.sources[i], float(sims[i])) for i in idx]

def _summarize_snippets(question: str, hits: List[Tuple[str, str, float]]) -> str:
    if not hits: 
        return ""
    text = "\n\n".join(h[0] for h in hits)
    # Try to extract concise facts (prices, times, days, emails, address)
    bullets = []

    # prices
    prices = re.findall(r"\$ ?\d+(?:\.\d{2})?", text)
    if prices:
        bullets.append("Pricing mentioned: " + ", ".join(sorted(set(prices))[:4]))

    # days/classes
    days = re.findall(r"\b(Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)\b", text, re.I)
    if days:
        bullets.append("Days referenced: " + ", ".join(sorted(set(d.title() for d in days))))

    # childcare/policy keywords
    if re.search(r"child\s*care|childcare", text, re.I):
        bullets.append("Childcare available (details in policy).")
    if re.search(r"cancel|freeze|late|policy|terms", text, re.I):
        bullets.append("See policies for freeze/cancel/late rules.")

    # fallback: first couple of short sentences matching keywords
    kws = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 3]
    if kws:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        picked = []
        for s in sentences:
            if any(k in s.lower() for k in kws):
                s = s.strip()
                if 15 <= len(s) <= 180:
                    picked.append(s)
            if len(picked) >= 3: break
        if picked:
            bullets.extend(picked)

    if not bullets:
        return ""
    return " • " + "\n • ".join(bullets)

def _answer_from_kb(question: str, min_sim: float = 0.15) -> str | None:
    kb: KB | None = st.session_state.get("kb_obj")
    if not kb:
        kb = _load_index_if_exists()
        if kb:
            st.session_state.kb_obj = kb
            st.session_state.kb_ready = True
    if not kb:
        return None
    hits = _retrieve(kb, question, k=4)
    if not hits or max(h[2] for h in hits) < min_sim:
        return None
    summary = _summarize_snippets(question, hits) or ""
    if st.session_state.show_sources and hits:
        seen = []
        for _, s, _ in hits:
            if s not in seen: seen.append(s)
        summary += ("\n\n" if summary else "") + "_sources: " + ", ".join(seen[:4])
    return summary or None

# ---------- Intents ----------
def wants_handoff(t: str) -> bool:
    return _wants_handoff_external(t)

def detect_intent(t: str) -> str | None:
    t = t.lower()
    if re.search(r"\b(member(ship)?|join|price|cost|plan|tier)\b", t): return "membership"
    if re.search(r"\b(class|schedule|time|what.*today|when.*class)\b", t): return "schedule"
    if wants_handoff(t): return "handoff"
    return None

# ---------- Calendly UI ----------
def show_manager_slots():
    st.chat_message("assistant").write("Absolutely — pick a time that works best for you:")
    tz = pytz.timezone(CALENDLY_TZ)
    now = dt.datetime.now(tz)
    start, end = now, now + dt.timedelta(days=7)

    if CALENDLY_EVENT_TYPE:
        try:
            slots = list_available_times(CALENDLY_EVENT_TYPE, start, end, CALENDLY_TZ)
        except Exception:
            slots = []
        if slots:
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
    # fallback
    if CALENDLY_URL:
        st.write(f'<a href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>', unsafe_allow_html=True)
    else:
        st.warning("Scheduler temporarily unavailable. Add CALENDLY_EVENT_TYPE or CALENDLY_URL in secrets.")

# ---------- Guided flows ----------
def start_membership_flow():
    st.session_state.flow = {"name": "membership", "stage": "freq"}
    st.chat_message("assistant").write(
        "Our monthly plans range by visit frequency and childcare.\n\n"
        "How often do you plan to come each week?",
    )
    st.chat_message("assistant").write("Choose one:")
    c1, c2, c3 = st.columns(3)
    with c1: st.button("~2x / week", key=f"m2", on_click=lambda: _set_flow_choice("~2"))
    with c2: st.button("~3x / week", key=f"m3", on_click=lambda: _set_flow_choice("~3"))
    with c3: st.button("Unlimited", key=f"mu", on_click=lambda: _set_flow_choice("unlimited"))

def _set_flow_choice(val: str):
    st.session_state.flow["choice"] = val
    st.session_state.flow["stage"] = "childcare"

def continue_membership_flow(user_text: str = ""):
    flow = st.session_state.flow
    if flow.get("stage") == "freq" and user_text:
        # accept natural answers too
        if re.search(r"unlimit|every|daily|any", user_text, re.I): _set_flow_choice("unlimited")
        elif re.search(r"3|three", user_text): _set_flow_choice("~3")
        else: _set_flow_choice("~2")

    if flow.get("stage") == "childcare":
        st.chat_message("assistant").write("Do you need childcare included?")
        c1, c2 = st.columns(2)
        with c1: st.button("Yes, include childcare", key="cc_yes", on_click=lambda: _finish_membership(True))
        with c2: st.button("No childcare", key="cc_no", on_click=lambda: _finish_membership(False))
        return

def _finish_membership(with_childcare: bool):
    choice = st.session_state.flow.get("choice", "~2")
    # Pull concise price hints from KB if possible
    kb_hint = _answer_from_kb("membership pricing childcare") or ""
    dollars = re.findall(r"\$ ?\d+(?:\.\d{2})?", kb_hint)
    price_hint = ""
    if dollars:
        uniq = sorted(set(dollars), key=lambda x: float(re.sub(r"[^\d.]", "", x)))
        if len(uniq) >= 2:
            price_hint = f"Prices typically range {uniq[0]}–{uniq[-1]}."
        else:
            price_hint = f"Typical price: {uniq[0]}."
    # Recommend
    if choice == "~2":
        rec = "Our **Sapphire**-style plan (about 2 classes/week)"
    elif choice == "~3":
        rec = "Our **Pearl**-style plan (about 3 classes/week)"
    else:
        rec = "Our **Diamond**-style plan (unlimited classes)"

    cc = " with **childcare add-on**" if with_childcare else ""
    msg = f"{rec}{cc} is usually the best fit. {price_hint or ''}".strip()

    st.chat_message("assistant").markdown(msg)
    st.chat_message("assistant").markdown(
        "Would you like to:\n\n"
        "• **See available times** to talk to our manager?\n"
        "• **Leave your info** and we’ll follow up?\n"
        "• Or keep asking questions."
    )
    st.session_state.flow = {}

def start_schedule_flow():
    st.session_state.flow = {"name": "schedule", "stage": "day"}
    st.chat_message("assistant").write("Great! What day do you want to work out? (e.g., Monday)")

def continue_schedule_flow(user_text: str):
    flow = st.session_state.flow
    if flow.get("stage") == "day":
        m = re.search(r"(mon|tue|wed|thu|fri|sat|sun)[a-z]*", user_text, re.I)
        if not m:
            st.chat_message("assistant").write("Please tell me a day (e.g., Monday).")
            return
        flow["day"] = m.group(0).title()
        flow["stage"] = "time"
        st.chat_message("assistant").write(f"Awesome — what time window on {flow['day']}? (e.g., 6–10am or 5–8pm)")
        return
    if flow.get("stage") == "time":
        # Try to extract hour(s)
        flow["time"] = user_text.strip()
        # Query KB for that day
        kb_ans = _answer_from_kb(f"{flow['day']} class schedule") or _answer_from_kb("class schedule")
        if not kb_ans:
            st.chat_message("assistant").write("I’m not sure. Would you like to speak with our manager?")
        else:
            # Pull lines containing that day
            lines = [ln for ln in kb_ans.splitlines() if flow["day"] in ln or re.search(r"\d", ln)]
            pretty = "\n".join("- " + ln.strip("•- ") for ln in lines[:8]) or kb_ans
            st.chat_message("assistant").markdown(f"Here’s what’s happening on **{flow['day']}** around **{flow['time']}**:\n\n{pretty}")
            st.chat_message("assistant").markdown("Need a different day or time? Ask away, or I can connect you with a person.")
        st.session_state.flow = {}

# ---------- Main handlers ----------
def handle_user_turn(user_text: str):
    st.chat_message("user").write(user_text)

    # Continue an active flow
    if st.session_state.flow.get("name") == "membership":
        continue_membership_flow(user_text)
        return
    if st.session_state.flow.get("name") == "schedule":
        continue_schedule_flow(user_text)
        return

    # Detect new intent
    intent = detect_intent(user_text)
    if intent == "handoff":
        show_manager_slots()
        return
    if intent == "membership":
        start_membership_flow()
        return
    if intent == "schedule":
        start_schedule_flow()
        return

    # Try KB
    ans = _answer_from_kb(user_text)
    if ans:
        st.chat_message("assistant").markdown(ans)
        _maybe_offer_lead_capture(user_text)
        return

    # Unknown → escalate
    st.chat_message("assistant").write(
        "I’m not sure. Would you like to speak with our manager?"
    )
    show_manager_slots()
    _show_text_and_feedback_options()

def _maybe_offer_lead_capture(user_text: str):
    if re.search(r"\b(trial|join|sign\s*up|membership|personal training|nutrition)\b", user_text, re.I):
        with st.expander("Leave your info for a follow-up (optional)"):
            with st.form(f"lead_{int(time.time())}"):
                name  = st.text_input("Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone (optional)")
                ok = st.checkbox("I agree to be contacted about my inquiry.")
                submitted = st.form_submit_button("Send")
                if submitted and name and email and ok:
                    try:
                        add_lead(name, email, phone, interest="From chat", source="web")
                        st.success("Thanks! We’ll follow up shortly.")
                    except Exception:
                        st.info("Saved locally. (Lead sheet not configured.)")

def _show_text_and_feedback_options():
    st.info("Prefer to text? Get a reply within ~30 minutes: **+1 (443) 975-9649**.")
    if FEEDBACK_FORM_URL:
        st.write(f'[Report a complaint / feedback / app issue]({FEEDBACK_FORM_URL})')

# ---------- Sidebar / Header ----------
with st.sidebar:
    st.toggle("Show sources", value=st.session_state.show_sources, key="show_sources")
    st.caption(f"KB cache: {DATA_DIR}")
    st.markdown("---")
    if not st.session_state.is_admin:
        with st.popover("Admin mode"):
            pw = st.text_input("Password", type="password", key="pw1")
            if st.button("Log in", key="login1"):
                if pw == ADMIN_PASSWORD:
                    st.session_state.is_admin = True
                    st.success("Admin mode enabled.")
                else:
                    st.error("Wrong password.")
    else:
        st.success("Admin mode")
        if st.button("Log out", key="logout1"):
            st.session_state.is_admin = False

# Top bar admin toggle (quick)
top_cols = st.columns([1,1,2,2,2])
with top_cols[0]:
    if st.toggle("Admin", value=st.session_state.is_admin, key="admin_toggle"):
        if not st.session_state.is_admin:
            pw = st.text_input("Admin password", type="password", key="pw_top")
            if pw == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.rerun()
        else:
            pass

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
    c1, c2 = st.columns(2)
    with c1:
        disabled = not uploaded
        if st.button("Build/Replace knowledge base", type="primary", use_container_width=True, disabled=disabled):
            with st.spinner("Indexing documents…"):
                kb = _build_index(
                    [io.BytesIO(f.read()) for f in uploaded],
                    [f.name for f in uploaded],
                )
                st.session_state.kb_obj = kb
                st.session_state.kb_ready = True
            st.success(f"Indexed {len(kb.chunks)} chunks from {len(set(kb.sources))} file(s).")
    with c2:
        if st.button("Load existing index from disk", use_container_width=True):
            kb = _load_index_if_exists()
            if kb:
                st.session_state.kb_obj = kb
                st.session_state.kb_ready = True
                st.success(f"Loaded {len(kb.chunks)} chunks from disk.")
            else:
                st.warning("No saved index found yet.")
    st.markdown("---")

# ---------- Replay chat history ----------
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

# ---------- Chat input ----------
user_msg = st.chat_input("Type your question (schedule, memberships, childcare, etc.)") or ""
if user_msg:
    st.session_state.chat_history.append(("user", user_msg))
    handle_user_turn(user_msg)

# ---------- Footer ----------
st.markdown('<div class="footer">Powered by <span class="brand">ZARI</span></div>', unsafe_allow_html=True)
