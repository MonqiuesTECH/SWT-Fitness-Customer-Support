# app.py — SWT Fitness Customer Support (Pro UI, free stack)
# - Polished UI (header + chips + bubbles) + "Powered by ZARI"
# - Admin toggle at top (login + PDF KB upload + knobs)
# - PDF/Text RAG (TF-IDF) with concise answers (no copy-paste walls)
# - Guided dialogs: Membership (price range → frequency → plan) & Schedule (day → time → classes)
# - Human handoff via Calendly (public link)
# - Optional lead capture to Google Sheets (leads.py), safe to omit

from __future__ import annotations

import os, io, re, json, time, textwrap, datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import streamlit as st
from pypdf import PdfReader
import pytz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ---------- Optional leads (safe no-op if file absent) ----------
try:
    from leads import add_lead  # expects add_lead(name, email, phone, interest, source)
except Exception:
    def add_lead(*args, **kwargs):
        pass  # no-op if not configured


# ================================================================
# Configuration (secrets with sane defaults)
# ================================================================
st.set_page_config(page_title="SWT Fitness — Customer Support", page_icon="💬", layout="wide")

STUDIO_NAME     = st.secrets.get("STUDIO_NAME", "SWT Fitness")
ADMIN_PASSWORD  = st.secrets.get("ADMIN_PASSWORD", "SWT_2025!manager")
UNKNOWN_MESSAGE = st.secrets.get("UNKNOWN_MESSAGE", "I'm not sure — would you like to speak with the manager?")
MIN_SIMILARITY  = float(st.secrets.get("MIN_SIMILARITY", 0.12))
CALENDLY_URL    = st.secrets.get("CALENDLY_URL", "").strip()
CALENDLY_TZ     = st.secrets.get("CALENDLY_TZ", "America/New_York")

# Branding (for Pro UI)
BRAND_COLOR     = st.secrets.get("BRAND_COLOR", "#0D7AFF")
AVATAR_LETTER   = st.secrets.get("AVATAR_LETTER", (STUDIO_NAME or "SWT")[0:1].upper())
AVATAR_BG       = st.secrets.get("AVATAR_BG", "#ffffff")

# Storage (Streamlit Cloud is writable at /tmp)
DATA_DIR  = st.secrets.get("DATA_DIR", "/tmp")
VEC_PATH  = os.path.join(DATA_DIR, "kb_vectorizer.joblib")
MAT_PATH  = os.path.join(DATA_DIR, "kb_matrix.joblib")
META_PATH = os.path.join(DATA_DIR, "kb_meta.json")

# ================================================================
# Session State
# ================================================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Tuple[str, str]] = []
if "flow" not in st.session_state:
    st.session_state.flow = None  # "membership" | "schedule" | None
if "slots" not in st.session_state:
    st.session_state.slots = {}   # store dialog slots (e.g., freq, day, time)
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

# ================================================================
# Pro UI CSS + Header (with "Powered by ZARI")
# ================================================================
PRO_UI_CSS = f"""
<style>
#MainMenu, header, footer {{ display: none !important; }}
section.main > div {{ padding-top: .5rem; }}
.block-container {{ padding-top: .75rem; max-width: 840px; }}

/* Brand header */
.z-header {{
  position: sticky; top: 0; z-index: 10;
  display:flex; align-items:center; justify-content:space-between; gap:.75rem;
  padding: .75rem 1rem; border-radius: 14px;
  background: linear-gradient(180deg, rgba(13,122,255,.10), rgba(13,122,255,.04));
  border: 1px solid rgba(13,122,255,.20);
  backdrop-filter: blur(4px);
  margin-bottom: .5rem;
}}
.z-left {{ display:flex; align-items:center; gap:.75rem; }}
.z-avatar {{
  width:32px; height:32px; border-radius:50%;
  display:grid; place-items:center; font-weight:800;
  color: {BRAND_COLOR}; background:{AVATAR_BG};
  box-shadow: 0 2px 8px rgba(0,0,0,.08);
}}
.z-title {{ font-weight:700; font-size:15px }}
.z-sub   {{ opacity:.7; font-size:13px }}

/* "Powered by ZARI" tag */
.z-tag {{
  display:inline-block; padding:6px 10px; border-radius:999px;
  border:1px solid rgba(13,122,255,.25); color:{BRAND_COLOR}; background:#fff;
  font: 700 12px/1 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  text-decoration:none;
}}

/* Message bubbles */
[data-testid="stChatMessage"] {{ background: transparent !important; }}
/* assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stMarkdownContainer"]) div[role="document"] {{
  border: 1px solid #ececec; border-radius: 14px; padding: 10px 12px;
  box-shadow: 0 6px 16px rgba(0,0,0,.05); background:#fff;
}}
/* user bubble */
[data-testid="stChatMessage"] div[role="document"]:has(.user-bubble) {{
  background: {BRAND_COLOR}; color: #fff; border: 0 !important;
}}
.user-bubble p {{ color:#fff !important; }}

/* Chips row */
.z-chips {{ display:flex; flex-wrap:wrap; gap:8px; margin: .25rem 0 .5rem; }}
.z-chip  {{ border:1px solid #e6e6e8; border-radius:999px; padding:8px 12px; background:#fff;
           cursor:pointer; font-size:13px; transition:all .15s ease; user-select:none; }}
.z-chip:hover {{ border-color:{BRAND_COLOR}; box-shadow:0 4px 12px rgba(13,122,255,.15); }}

/* Top admin bar */
.z-adminbar {{ display:flex; align-items:center; gap:12px; margin: 6px 2px 10px; }}
.z-foot {{ text-align:center; font-size:12px; opacity:.55; margin:10px 0 6px; }}
</style>
"""
st.markdown(PRO_UI_CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="z-header">
      <div class="z-left">
        <div class="z-avatar">{AVATAR_LETTER}</div>
        <div>
          <div class="z-title">{STUDIO_NAME} — Customer Support</div>
          <div class="z-sub">Ask about memberships, schedule, childcare, promotions, and more.</div>
        </div>
      </div>
      <div class="z-right">
        <a class="z-tag" href="https://zari.ai" target="_blank" rel="noopener">Powered by ZARI</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top admin toggle / login
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    admin_toggle = st.toggle("Admin mode", value=st.session_state.is_admin, key="admin_toggle", help="Login to upload PDFs and tweak settings")
with c2:
    pass
with c3:
    st.caption("")

if admin_toggle and not st.session_state.is_admin:
    pw = st.text_input("Enter admin password", type="password")
    if st.button("Log in", type="primary"):
        if pw == ADMIN_PASSWORD:
            st.session_state.is_admin = True
            st.success("Admin mode enabled.")
        else:
            st.error("Wrong password.")
elif not admin_toggle and st.session_state.is_admin:
    st.session_state.is_admin = False
    st.info("Admin mode disabled.")

# ================================================================
# KB: Chunk / Index / Cache
# ================================================================
@dataclass
class KB:
    vectorizer: TfidfVectorizer
    matrix: Any
    chunks: List[str]
    sources: List[str]

def _chunk(text: str, chunk_size: int = 500, overlap: int = 120) -> List[str]:
    words = text.split()
    if not words: return []
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += max(1, (chunk_size - overlap))
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
        for ch in _chunk(txt, 520, 140):
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

@st.cache_resource(show_spinner=False)
def cached_load_index() -> KB | None:
    try:
        if not (os.path.exists(VEC_PATH) and os.path.exists(MAT_PATH) and os.path.exists(META_PATH)):
            return None
        vectorizer = joblib.load(VEC_PATH)
        matrix = joblib.load(MAT_PATH)
        meta = json.load(open(META_PATH))
        return KB(vectorizer=vectorizer, matrix=matrix,
                  chunks=meta["chunks"], sources=meta["sources"])
    except Exception:
        return None

# Warm on boot
if "kb_obj" not in st.session_state:
    st.session_state.kb_obj = cached_load_index()
    st.session_state.kb_ready = bool(st.session_state.kb_obj)

# ================================================================
# Retrieval + Answer composition (concise, no wall-of-text)
# ================================================================
def _retrieve(kb: KB, query: str, k: int = 4) -> List[Tuple[str, str, float]]:
    qv = kb.vectorizer.transform([query])
    sims = cosine_similarity(qv, kb.matrix).ravel()
    idx = np.argsort(-sims)[:k]
    out = [(kb.chunks[i], kb.sources[i], float(sims[i])) for i in idx]
    return out

def _best_snippets(question: str, hits: List[Tuple[str, str, float]], limit_chars=600) -> str:
    # Pull short sentences relevant to question keywords
    kws = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 3]
    sentences = []
    for chunk, _, _ in hits:
        for s in re.split(r"(?<=[.!?])\s+", chunk):
            ls = s.lower()
            if any(k in ls for k in kws):
                sentences.append(s.strip())
    if not sentences and hits:
        sentences = re.split(r"(?<=[.!?])\s+", hits[0][0])[:3]
    ans = " ".join(sentences)
    return textwrap.shorten(ans.strip(), width=limit_chars, placeholder="…")

def _compose_answer(question: str, kb: KB) -> Tuple[str, List[str], float]:
    hits = _retrieve(kb, question, k=4)
    if not hits:
        return "", [], 0.0
    best_sim = max(h[2] for h in hits)
    concise = _best_snippets(question, hits)
    srcs = []
    for _, src, _ in hits:
        if src not in srcs:
            srcs.append(src)
    return concise, srcs, best_sim

# ================================================================
# Intents & Dialog helpers
# ================================================================
def wants_handoff(text: str) -> bool:
    return bool(re.search(r"\b(speak|talk|human|person|manager|call|phone)\b", text, re.I))

def parse_frequency(text: str) -> str | None:
    t = text.lower()
    if "unlimited" in t: return "unlimited"
    words_map = {
        r"\b(one|once|1)\b": "1",
        r"\b(two|twice|2)\b": "2",
        r"\b(three|3)\b": "3",
        r"\b(four|4)\b": "4",
        r"\b(five|5)\b": "5",
        r"\b(six|6)\b": "6",
        r"\b(seven|7)\b": "7",
    }
    for pat, val in words_map.items():
        if re.search(pat, t):
            return val
    return None

def price_range_from_kb(kb: KB) -> Tuple[str, str] | None:
    # Look for $ amounts across chunks
    amounts = []
    for ch in kb.chunks:
        for m in re.findall(r"\$\s*([0-9]+(?:\.[0-9]{2})?)", ch):
            try: amounts.append(float(m))
            except: pass
    if not amounts: return None
    lo, hi = min(amounts), max(amounts)
    # Round "nice"
    return (f"${lo:,.0f}" if lo.is_integer() else f"${lo:,.2f}",
            f"${hi:,.0f}" if hi.is_integer() else f"${hi:,.2f}")

def recommend_plan_by_freq(kb: KB, freq: str) -> str:
    # Heuristic: pick sentences that mention freq or unlimited
    q = "membership plan unlimited" if freq == "unlimited" else f"membership {freq} times per week plan"
    hits = _retrieve(kb, q, k=4)
    if not hits:
        if freq == "unlimited":
            return "Unlimited plan"
        return f"{freq}x/week plan"
    # extract a likely plan/tier name from hits
    text = " ".join(h[0] for h in hits)
    # look for capitalized tier-like words (Pearl, Gold, Unlimited, Basic, etc.)
    tiers = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s*&\s*[A-Z][a-zA-Z]+)?)\b", text)
    # filter noise
    tiers = [t for t in tiers if t.lower() not in {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}]
    if tiers:
        return tiers[0]
    return "Recommended plan"

def parse_day(text: str) -> str | None:
    days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday","today","tomorrow"]
    for d in days:
        if re.search(rf"\b{d}\b", text.lower()):
            return d
    return None

def parse_timewindow(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(morning|am|a\.m\.)\b", t): return "morning"
    if re.search(r"\b(afternoon)\b", t): return "afternoon"
    if re.search(r"\b(evening|pm|p\.m\.)\b", t): return "evening"
    # specific times (e.g., 6:00, 6pm)
    m = re.search(r"\b([0-1]?\d)(?::([0-5]\d))?\s*(am|pm)?\b", t)
    if m:
        hr = int(m.group(1)); mer = (m.group(3) or "").lower()
        if mer == "am" and 5 <= hr <= 11: return "morning"
        if mer == "pm" and 12 <= hr <= 16: return "afternoon"
        if mer == "pm" and 16 <= hr <= 22: return "evening"
    return None

def show_manager_link():
    if CALENDLY_URL:
        st.chat_message("assistant").write(
            f'<a href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>',
            unsafe_allow_html=True
        )
    else:
        st.chat_message("assistant").write("Our manager will follow up shortly.")

# ================================================================
# Answering + Dialog Controller
# ================================================================
def answer_question(question: str) -> str:
    kb: KB | None = st.session_state.get("kb_obj") or cached_load_index()
    if kb and not st.session_state.get("kb_obj"):
        st.session_state.kb_obj = kb
        st.session_state.kb_ready = True
    if not kb:
        return "The knowledge base isn’t loaded yet. Go to Admin → upload PDFs, then ask again."

    concise, _, best_sim = _compose_answer(question, kb)
    if not concise or best_sim < MIN_SIMILARITY:
        return f"{UNKNOWN_MESSAGE}"
    return concise

def handle_turn(user_text: str):
    if not user_text: return
    st.chat_message("user").markdown(f"<div class='user-bubble'>{user_text}</div>", unsafe_allow_html=True)

    # Handoff intent always wins
    if wants_handoff(user_text):
        st.chat_message("assistant").write("Absolutely — pick a time that works best for you:")
        show_manager_link()
        return

    # Membership dialog
    if st.session_state.flow == "membership" or re.search(r"\b(membership|join|sign\s*up)\b", user_text, re.I):
        st.session_state.flow = "membership"
        kb = st.session_state.get("kb_obj")
        pr = price_range_from_kb(kb) if kb else None
        freq = parse_frequency(user_text)
        if "freq" not in st.session_state.slots and not freq:
            if pr:
                lo, hi = pr
                st.chat_message("assistant").write(
                    f"Our memberships range from **{lo}–{hi}**. How often do you want to work out each week? (2 / 3 / unlimited)"
                )
            else:
                st.chat_message("assistant").write(
                    "How often do you want to work out each week? (2 / 3 / unlimited)"
                )
            st.session_state.slots["freq"] = None
            return
        if freq:
            st.session_state.slots["freq"] = freq

        if st.session_state.slots.get("freq"):
            plan = recommend_plan_by_freq(kb, st.session_state.slots["freq"])
            # Compose concise recommendation; avoid raw copy/paste
            extra = answer_question("membership benefits and included features")
            msg = f"I recommend the **{plan}** based on going **{st.session_state.slots['freq']}x/week**."
            if extra and extra != UNKNOWN_MESSAGE:
                msg += f" {textwrap.shorten(extra, 160, placeholder='…')}"
            st.chat_message("assistant").write(msg)
            st.chat_message("assistant").write("Want to book a quick consult with our manager?")
            show_manager_link()
            # Reset flow after recommendation
            st.session_state.flow = None
            st.session_state.slots = {}
            return

    # Schedule dialog
    if st.session_state.flow == "schedule" or re.search(r"\b(schedule|class|spin|barre|circuit|yoga|training)\b", user_text, re.I):
        st.session_state.flow = "schedule"
        day = parse_day(user_text) or st.session_state.slots.get("day")
        timewin = parse_timewindow(user_text) or st.session_state.slots.get("timewin")
        if not day:
            st.chat_message("assistant").write("Okay — what **day** would you like to work out?")
            st.session_state.slots["day"] = None
            return
        st.session_state.slots["day"] = day
        if not timewin:
            st.chat_message("assistant").write("Great — what **time** works? (morning / afternoon / evening or a time like 6:00am)")
            st.session_state.slots["timewin"] = None
            return
        st.session_state.slots["timewin"] = timewin

        kb = st.session_state.get("kb_obj")
        q = f"classes on {day} {timewin}"
        ans = answer_question(q)
        if ans == UNKNOWN_MESSAGE:
            ans = answer_question(f"{day} class schedule {timewin}")
        if ans == UNKNOWN_MESSAGE:
            st.chat_message("assistant").write(UNKNOWN_MESSAGE)
            show_manager_link()
        else:
            # Render concise list-like answer
            st.chat_message("assistant").write(ans)
            st.chat_message("assistant").write("Want to speak with our manager to pick a class?")
            show_manager_link()
        # Reset flow
        st.session_state.flow = None
        st.session_state.slots = {}
        return

    # General Q&A
    ans = answer_question(user_text)
    st.chat_message("assistant").write(ans)
    if ans == UNKNOWN_MESSAGE:
        show_manager_link()

    # Lead capture nudge on buying intent
    if re.search(r"\b(trial|join|sign\s*up|membership|personal training|nutrition|consult)\b", user_text, re.I):
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


# ================================================================
# Quick-reply chips
# ================================================================
def quick_chips():
    st.markdown('<div class="z-chips">', unsafe_allow_html=True)
    cols = st.columns(4)
    chips = [
        ("Membership prices", "How much is a membership?"),
        ("What’s on today", "What classes are available today?"),
        ("Childcare", "Do you offer childcare?"),
        ("Speak to a person", "I want to speak with someone"),
    ]
    for i, (label, text) in enumerate(chips):
        if cols[i].button(label, key=f"chip_{i}", use_container_width=True):
            st.session_state.chat_history.append(("user", text))
            st.session_state["__chip_sent"] = text
    st.markdown('</div>', unsafe_allow_html=True)


# ================================================================
# Admin Panel (Top)
# ================================================================
if st.session_state.is_admin:
    with st.expander("Admin · Load / replace knowledge base (PDF/Text) · Settings", expanded=False):
        uploaded = st.file_uploader(
            "Upload one or more PDFs (pricing, schedule, policies). Use text-based PDFs (not scans).",
            type=["pdf"], accept_multiple_files=True
        )
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            if st.button("Build/Replace knowledge base", type="primary", disabled=(not uploaded)):
                with st.spinner("Indexing documents…"):
                    kb = _build_index(
                        from_files=[io.BytesIO(f.read()) for f in uploaded],
                        filenames=[f.name for f in uploaded],
                    )
                    st.session_state.kb_obj = kb
                    st.session_state.kb_ready = True
                st.success(f"Indexed {len(kb.chunks)} chunks from {len(set(kb.sources))} file(s).")
        with colB:
            if st.button("Load existing index from disk"):
                kb = cached_load_index()
                if kb:
                    st.session_state.kb_obj = kb
                    st.session_state.kb_ready = True
                    st.success(f"Loaded {len(kb.chunks)} chunks from disk.")
                else:
                    st.warning("No saved index found yet.")
        with colC:
            new_thr = st.number_input("Similarity threshold", min_value=0.05, max_value=0.50, step=0.01, value=float(MIN_SIMILARITY))
            if st.button("Apply threshold"):
                st.success(f"Using MIN_SIMILARITY={new_thr:.2f} (until app restarts)")
                # store per-session (secrets are immutable at runtime)
                st.session_state.MIN_SIMILARITY = float(new_thr)

# Use session override if admin adjusted threshold
if "MIN_SIMILARITY" in st.session_state:
    MIN_SIMILARITY = float(st.session_state.MIN_SIMILARITY)

# ================================================================
# Top chips + Transcript + Input
# ================================================================
quick_chips()

# Replay previous messages
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.chat_message("assistant").write(msg)

# Handle chip click as a turn
if st.session_state.get("__chip_sent"):
    handle_turn(st.session_state.pop("__chip_sent"))

# Normal input
user_msg = st.chat_input("Type your question (schedule, memberships, childcare, etc.)") or ""
if user_msg:
    st.session_state.chat_history.append(("user", user_msg))
    handle_turn(user_msg)

# Footer watermark
st.markdown('<div class="z-foot">Powered by ZARI</div>', unsafe_allow_html=True)
