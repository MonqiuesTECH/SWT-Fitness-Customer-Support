# app.py — SWT Fitness Customer Support (dialog + concise answers)
# - PDF/Text KB (pypdf + TF-IDF)
# - Dialog flows:
#     • Membership: range -> ask frequency -> recommend plan
#     • Schedule: ask day -> ask time -> show classes
# - If unsure: "I'm not sure — would you like to speak with the manager?" + Calendly
# - Admin upload + index
# - Lead capture gate (uses leads.py; safe no-op if not configured)
from __future__ import annotations

import io, json, re, time, textwrap, datetime as dt, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any, Dict

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
from leads import add_lead  # optional

# -------------------- Config & Secrets --------------------
st.set_page_config(page_title="SWT Fitness Customer Support", page_icon="💬", layout="wide")

STUDIO_NAME    = st.secrets.get("STUDIO_NAME", "SWT Fitness")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin")

CALENDLY_EVENT_TYPE = st.secrets.get("CALENDLY_EVENT_TYPE", "").strip()  # API URI (optional)
CALENDLY_URL        = st.secrets.get("CALENDLY_URL", "").strip()          # public link (fallback)
CALENDLY_TZ         = st.secrets.get("CALENDLY_TZ", "America/New_York")
UNKNOWN_MESSAGE     = st.secrets.get(
    "UNKNOWN_MESSAGE",
    "I'm not sure — would you like to speak with the manager?"
)

# Confidence threshold for generic QA
MIN_SIMILARITY = float(st.secrets.get("MIN_SIMILARITY", 0.12))

# Writable storage (Streamlit free tier)
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
# dialog state
ss.setdefault("pending_intent", None)           # "membership" | "schedule" | None
ss.setdefault("slots", {})                      # e.g., {"frequency": 3, "day": "monday", "time_pref": "evening"}
ss.setdefault("membership_map", {})             # cached tiers/prices
ss.setdefault("schedule_index", {})             # cached day -> list[(time, class)]

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
    idx = np.argsort(-sims)[:k]
    return [(kb.chunks[i], kb.sources[i], float(sims[i])) for i in idx]

def _full_text(kb: KB) -> str:
    return "\n".join(kb.chunks)

# -------------------- Parsers: Memberships & Schedule --------------------
def extract_memberships(kb: KB) -> Dict[str, Dict[str, str]]:
    """
    Returns {"Sapphire": {"price":"59.99","childcare":"84.99","classes_per_week":"2"}, ...}
    Best-effort, from all KB text.
    """
    text = _full_text(kb)
    text = re.sub(r'\s+', ' ', text)

    tiers = {"Sapphire":["Sapphire","Saphire"], "Pearl":["Pearl"], "Diamond":["Diamond"]}
    out: Dict[str, Dict[str, str]] = {}

    for tier, aliases in tiers.items():
        m = None
        for alias in aliases:
            m = re.search(rf"{alias}[^$]*\$\s*([0-9]+(?:\.[0-9]{{2}})?)", text, re.I)
            if m: break
        if not m: continue
        price = m.group(1)

        # classes/week
        c = None
        for alias in aliases:
            c = re.search(rf"{alias}.*?(\d+)\s*classes?\s*/\s*week", text, re.I)
            if c: break

        # childcare near alias
        cc = None
        for alias in aliases:
            cc = re.search(rf"{alias}[^$]{{0,120}}(?:with|w/?|w)\s*child\s*care[^$]*\$\s*([0-9]+(?:\.[0-9]{{2}})?)", text, re.I)
            if cc: break

        out[tier] = {
            "price": price,
            "childcare": cc.group(1) if cc else "",
            "classes_per_week": c.group(1) if c else ""
        }
    return out

def build_schedule_index(kb: KB) -> Dict[str, List[Tuple[str,str]]]:
    """
    Very simple extractor: for each weekday, grab nearby time + class pairs.
    Returns {"monday":[("5:30AM","SWT RIP"), ...], ...}
    """
    text = _full_text(kb)
    text = text.replace("–","-")
    days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    idx: Dict[str, List[Tuple[str,str]]] = {d:[] for d in days}

    for day in days:
        for m in re.finditer(rf"{day}\b(.{{0,800}})", text, re.I):
            seg = m.group(0)
            # find lines like "5:30AM X", "6:00 AM SPIN", "6:00am - SPIN"
            for t, cls in re.findall(r"(\b\d{1,2}:\d{2}\s*(?:am|pm)\b)\s*[-–]?\s*([A-Za-z][^\n•\r]{2,50})", seg, re.I):
                cls = re.sub(r"[\(\)•\-\u2022]+","",cls).strip()
                idx[day].append((t.upper().replace(" ",""), cls))
    # de-dup shallowly
    for d in days:
        seen=set(); clean=[]
        for t,c in idx[d]:
            key=(t,c.lower())
            if key in seen: continue
            seen.add(key); clean.append((t,c))
        idx[d]=clean[:20]
    return idx

# -------------------- Concise QA Fallback --------------------
def _concise_answer(question: str, hits: List[Tuple[str, str, float]]) -> str | None:
    """Short, non-copy-paste fallback for general Qs."""
    q = question.lower()
    text = "\n".join(h[0] for h in hits)
    text = re.sub(r'\S+@\S+|https?://\S+',' ', text)
    text = re.sub(r'\s+',' ', text)

    # simple class list
    if re.search(r"\bclass|schedule|spin|barre|zumba|circuit|trainer|dance\b", q):
        return "Popular classes: Trainer Takeover, Silver Belles, Circuit Training, Barre, Spin, Dance & Burn."

    # hours
    if re.search(r"\bhours?|open|close|staff(ed)?\b", q):
        rng = re.findall(r'(\d{1,2}:\d{2}\s*(?:a|p)m)\s*-\s*(\d{1,2}:\d{2}\s*(?:a|p)m)', text, re.I)
        if rng:
            uniq = []
            for a,b in rng:
                s=f"{a}-{b}".replace(" ","")
                if s not in uniq: uniq.append(s)
            return f"Staffed hours: {', '.join(uniq[:4])}. Members have 24/7 access."
    # tiny sentence pick
    sents = re.split(r"(?<=[.!?])\s+", text)
    kws = set(re.findall(r"[a-z]{4,}", q))
    for s in sents:
        s2 = re.sub(r'\s+',' ', s).strip()
        if 20 <= len(s2) <= 160 and any(k in s2.lower() for k in kws):
            return s2
    return None

# -------------------- Intent detection --------------------
def detect_intent(text: str) -> str | None:
    t = text.lower()
    if wants_handoff(text): return "handoff"
    if re.search(r"\b(member|price|cost|join|how much)\b", t): return "membership"
    if re.search(r"\b(schedule|class|spin|barre|zumba|circuit|trainer|yoga|time|what time)\b", t): return "schedule"
    return None

# -------------------- Calendly --------------------
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
            t = dt.datetime.fromisoformat(s["start_time"].replace("Z","+00:00")).astimezone(tz)
            by_day[t.strftime("%A %b %d")].append((t, s["scheduling_url"]))
        for day, entries in sorted(by_day.items(), key=lambda kv: kv[1][0][0]):
            with st.expander(day, expanded=len(by_day)==1):
                for t, url in entries:
                    label = t.strftime("%-I:%M %p")
                    st.write(
                        f'<a href="{url}" target="_blank" rel="noopener" '
                        f'style="display:inline-block;margin:6px 8px;padding:8px 12px;'
                        f'border-radius:10px;border:1px solid #ddd;text-decoration:none;">'
                        f'Book {label}</a>', unsafe_allow_html=True)
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
st.title(f"{STUDIO_NAME} Customer Support")
st.caption("Ask about classes, schedules, childcare, pricing, promotions, and more.")

# -------------------- Admin Panel --------------------
if ss.is_admin:
    st.subheader("Admin · Load / replace knowledge base (PDF/Text)")
    uploaded = st.file_uploader("Upload one or more PDFs (pricing, policy, schedule).", type=["pdf"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Build/Replace knowledge base", type="primary", use_container_width=True, disabled=(not uploaded)):
            with st.spinner("Indexing documents…"):
                kb = _build_index([io.BytesIO(f.read()) for f in uploaded], [f.name for f in uploaded])
                ss.kb_obj = kb
                ss.kb_ready = True
                # refresh caches
                ss.membership_map = extract_memberships(kb)
                ss.schedule_index = build_schedule_index(kb)
            st.success(f"Indexed {len(kb.chunks)} chunks from {len(set(kb.sources))} file(s).")
    with c2:
        if st.button("Load existing index from disk", use_container_width=True):
            kb = _load_index_if_exists()
            if kb:
                ss.kb_obj = kb
                ss.kb_ready = True
                ss.membership_map = extract_memberships(kb)
                ss.schedule_index = build_schedule_index(kb)
                st.success(f"Loaded {len(kb.chunks)} chunks from disk.")
            else:
                st.warning("No saved index found yet.")
    st.markdown("---")

# -------------------- Lead Gate --------------------
def lead_gate():
    if ss.lead_captured: return
    st.chat_message("assistant").write("Can I get your **name**, **email**, and **phone** so we can follow up?")
    with st.form("lead_gate_form", clear_on_submit=False, enter_to_submit=True):
        c1, c2 = st.columns(2)
        name  = c1.text_input("Name *")
        email = c2.text_input("Email *")
        phone = st.text_input("Phone (optional)")
        agree = st.checkbox("I agree to be contacted about my inquiry.")
        colA, colB = st.columns([1,1])
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
                    st.success("Thanks! We’ll follow up if needed." if ok else "Saved locally. We’ll sync later.")
                except Exception:
                    st.info("Saved locally. (Lead store not configured yet.)")
                ss.lead_captured = True
                ss.lead_profile = {"name": name, "email": email, "phone": phone}
        if skip and not ss.lead_captured:
            ss.lead_captured = True
            ss.lead_profile = {}
            st.info("No problem — you can still ask questions anytime.")

# -------------------- Dialog Flows --------------------
def membership_flow(user_text: str, kb: KB):
    # Ensure membership map cached
    if not ss.membership_map:
        ss.membership_map = extract_memberships(kb)

    mm = ss.membership_map
    prices = [float(v["price"]) for v in mm.values() if v.get("price")]
    if not prices:
        st.chat_message("assistant").write(UNKNOWN_MESSAGE); show_manager_slots(); return

    lo, hi = min(prices), max(prices)
    # If we don't have frequency yet, ask
    if "frequency" not in ss.slots:
        st.chat_message("assistant").write(f"Our memberships range from ${lo:.2f}–${hi:.2f}. How often do you want to go each week? (2 / 3 / unlimited)")
        ss.pending_intent = "membership"
        return

    # We have frequency → recommend
    freq = ss.slots.get("frequency")
    # map frequencies to tiers
    choice = None
    if isinstance(freq, (int,float)):
        if freq <= 2: choice = "Sapphire"
        elif freq == 3: choice = "Pearl"
        else: choice = "Diamond"
    else:
        if re.search(r"unlimited|every\s*day|daily", str(freq), re.I): choice = "Diamond"

    if not choice or choice not in mm:
        st.chat_message("assistant").write(UNKNOWN_MESSAGE); show_manager_slots(); return

    price = mm[choice].get("price","")
    cw    = mm[choice].get("classes_per_week","")
    cc    = mm[choice].get("childcare","")
    msg = f"{choice}: ${price}"
    if cw: msg += f" ({cw}/week)"
    if cc: msg += f"; ${cc} with childcare"
    st.chat_message("assistant").write(f"I’d recommend **{msg}**. Want me to help you book a quick call to get started?")
    ss.pending_intent = None
    ss.slots.pop("frequency", None)

def _norm_day(txt: str) -> str | None:
    t = txt.lower()
    for d in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]:
        if d in t: return d
    return None

def _time_pref(txt: str) -> str | None:
    t = txt.lower()
    if re.search(r"\b(morning|am)\b", t): return "morning"
    if re.search(r"\b(afternoon)\b", t): return "afternoon"
    if re.search(r"\b(evening|pm|night)\b", t): return "evening"
    m = re.search(r"\b(\d{1,2}(:\d{2})?\s*(am|pm))\b", t)
    if m: return m.group(1).upper().replace(" ","")
    return None

def schedule_flow(user_text: str, kb: KB):
    # Ensure schedule index cached
    if not ss.schedule_index:
        ss.schedule_index = build_schedule_index(kb)

    # Collect slots progressively
    day = ss.slots.get("day") or _norm_day(user_text)
    if not day:
        st.chat_message("assistant").write("Okay — what **day** would you like to work out?")
        ss.pending_intent = "schedule"; return
    ss.slots["day"] = day

    tp = ss.slots.get("time_pref") or _time_pref(user_text)
    if not tp:
        st.chat_message("assistant").write("Great — what **time** works? (morning / afternoon / evening or a time like 6:00am)")
        ss.pending_intent = "schedule"; return
    ss.slots["time_pref"] = tp

    # Retrieve classes for that day/time
    entries = ss.schedule_index.get(day, [])
    if not entries:
        st.chat_message("assistant").write(UNKNOWN_MESSAGE); show_manager_slots(); ss.pending_intent=None; ss.slots.clear(); return

    def bucket(tstr: str) -> str:
        # tstr like 5:30AM
        hh = int(re.match(r"(\d{1,2})", tstr).group(1))
        is_pm = "PM" in tstr.upper()
        hour24 = (hh % 12) + (12 if is_pm else 0)
        if 5 <= hour24 < 12: return "morning"
        if 12 <= hour24 < 17: return "afternoon"
        return "evening"

    results = []
    if tp in ["morning","afternoon","evening"]:
        results = [(t,c) for (t,c) in entries if bucket(t)==tp]
    else:
        # specific time substring match
        results = [(t,c) for (t,c) in entries if tp in t]

    if not results:
        # fall back to show a few for that day
        results = entries[:5]

    lines = [f"{t} — {c}" for t,c in results[:6]]
    st.chat_message("assistant").write(f"**{day.capitalize()}** options:\n- " + "\n- ".join(lines))
    ss.pending_intent=None; ss.slots.clear()

# -------------------- Generic QA --------------------
def qa_answer(question: str) -> Tuple[str, bool]:
    kb: KB | None = ss.get("kb_obj")
    if not kb:
        kb = _load_index_if_exists()
        if kb:
            ss.kb_obj = kb; ss.kb_ready = True
            ss.membership_map = extract_memberships(kb)
            ss.schedule_index = build_schedule_index(kb)
    if not kb:
        return "The knowledge base isn’t loaded yet. Go to Admin → upload PDFs, then ask again.", False

    hits = _retrieve(kb, question, k=4)
    if not hits: return UNKNOWN_MESSAGE, False

    max_sim = max(h[2] for h in hits)
    ans = _concise_answer(question, hits)
    if not ans or max_sim < MIN_SIMILARITY:
        return UNKNOWN_MESSAGE, False

    if ss.show_sources:
        uniq=[]; 
        for _,src,_ in hits:
            if src not in uniq: uniq.append(src)
        if uniq: ans += "\n\n_sources: " + ", ".join(uniq[:4])
    return textwrap.shorten(ans, width=300, placeholder="…"), True

# -------------------- Turn Handler --------------------
def handle_turn(user_text: str):
    if not user_text: return
    st.chat_message("user").write(user_text)

    # If user explicitly asks for human
    if wants_handoff(user_text):
        show_manager_slots(); return

    kb = ss.get("kb_obj") or _load_index_if_exists()
    if kb and "kb_obj" not in ss: ss.kb_obj = kb

    # Continue an in-flight dialog first
    if ss.pending_intent == "membership":
        # try to capture frequency
        t = user_text.lower()
        freq = None
        if re.search(r"\b2|two|twice\b", t): freq = 2
        elif re.search(r"\b3|three\b", t): freq = 3
        elif re.search(r"\bunlimited|every\s*day|daily\b", t): freq = "unlimited"
        if freq is not None:
            ss.slots["frequency"] = freq
        membership_flow(user_text, kb)  # will complete or ask again
        return

    if ss.pending_intent == "schedule":
        # try to capture day/time
        d = _norm_day(user_text); 
        if d: ss.slots["day"] = d
        tp = _time_pref(user_text)
        if tp: ss.slots["time_pref"] = tp
        schedule_flow(user_text, kb); 
        return

    # New turn → detect intent
    intent = detect_intent(user_text)
    if intent == "membership":
        membership_flow(user_text, kb); return
    if intent == "schedule":
        schedule_flow(user_text, kb); return

    # Otherwise generic QA
    ans, ok = qa_answer(user_text)
    if ok:
        st.chat_message("assistant").write(ans)
    else:
        st.chat_message("assistant").write(UNKNOWN_MESSAGE)
        show_manager_slots()

    # lead hook
    if re.search(r"\b(trial|join|sign\s*up|membership|personal training|nutrition)\b", user_text, re.I):
        with st.expander("Leave your info for a follow-up (optional)"):
            with st.form(f"lead_{int(time.time())}"):
                name  = st.text_input("Name", value=ss.lead_profile.get("name",""))
                email = st.text_input("Email", value=ss.lead_profile.get("email",""))
                phone = st.text_input("Phone (optional)", value=ss.lead_profile.get("phone",""))
                submitted = st.form_submit_button("Send to team")
                if submitted and name and email:
                    try:
                        ok = add_lead(name, email, phone, interest="From chat", source="web")
                        st.success("Thanks! We’ll follow up shortly." if ok else "Saved locally. (Lead sheet not configured.)")
                    except Exception:
                        st.info("Saved locally. (Lead store not configured.)")

# -------------------- Chat UI --------------------
for role, msg in ss.chat_history:
    st.chat_message(role).write(msg)

# Ask for contact once
def lead_gate():
    if ss.lead_captured: return
    st.chat_message("assistant").write("Can I get your **name**, **email**, and **phone** so we can follow up?")
    with st.form("lead_gate_form", clear_on_submit=False, enter_to_submit=True):
        c1, c2 = st.columns(2)
        name  = c1.text_input("Name *")
        email = c2.text_input("Email *")
        phone = st.text_input("Phone (optional)")
        agree = st.checkbox("I agree to be contacted about my inquiry.")
        colA, colB = st.columns([1,1])
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
                    st.success("Thanks! You’re all set." if ok else "Saved locally. We’ll sync later.")
                except Exception:
                    st.info("Saved locally. (Lead store not configured yet.)")
                ss.lead_captured = True
                ss.lead_profile = {"name": name, "email": email, "phone": phone}
        if skip and not ss.lead_captured:
            ss.lead_captured = True
            ss.lead_profile = {}
            st.info("No problem — you can still ask questions anytime.")

lead_gate()

user_msg = st.chat_input("Type your question (schedule, memberships, childcare, etc.)") or ""
if user_msg:
    ss.chat_history.append(("user", user_msg))
    handle_turn(user_msg)
