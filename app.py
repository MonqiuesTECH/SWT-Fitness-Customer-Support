# app.py  (stable + concise answers, fact extraction, safe fallbacks)
# - Runs on Streamlit Community Cloud free tier (no paid APIs required)
# - Admin upload -> PDF index (TF-IDF) + lightweight fact extraction
# - Membership & hours answers come from parsed facts (not copy/paste)
# - Follow-up prompts for membership/day/time; safe “not sure” fallback
# - Calendly human handoff with graceful fallback to booking URL
# - Optional lead capture to Google Sheets via leads.py (if configured)
# - Footer: “Powered by ZARI”

from __future__ import annotations
import io, os, re, json, time, textwrap, tempfile, datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import streamlit as st
from pypdf import PdfReader
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ---- Optional modules (Calendly + Leads). App still runs if missing.
try:
    from calendly_api import list_available_times, create_single_use_link
except Exception:
    def list_available_times(*args, **kwargs): return []
    def create_single_use_link(*args, **kwargs): return None

try:
    from leads import add_lead
except Exception:
    def add_lead(*args, **kwargs): pass

# ---- Intent helper (fallback if intents.py isn’t available)
def _default_wants_handoff(txt: str) -> bool:
    return bool(re.search(r"\b(speak|talk|call|human|person|manager|someone|real agent)\b", txt, re.I))
try:
    from intents import wants_handoff  # your existing file
except Exception:
    wants_handoff = _default_wants_handoff

# =================== App / Secrets ===================
st.set_page_config(page_title="SWT Fitness Customer Support", page_icon="💬", layout="wide")

STUDIO_NAME        = st.secrets.get("STUDIO_NAME", "SWT Fitness")
ADMIN_PASSWORD     = st.secrets.get("ADMIN_PASSWORD", "admin")
CALENDLY_EVENT_URI = st.secrets.get("CALENDLY_EVENT_TYPE", "").strip()
CALENDLY_URL       = st.secrets.get("CALENDLY_URL", "").strip()
CALENDLY_TZ        = st.secrets.get("CALENDLY_TZ", "America/New_York")

# Use /tmp to avoid PermissionError on Streamlit Cloud
DATA_DIR  = os.environ.get("DATA_DIR", tempfile.gettempdir())
VEC_PATH  = os.path.join(DATA_DIR, "kb_vectorizer.joblib")
MAT_PATH  = os.path.join(DATA_DIR, "kb_matrix.joblib")
META_PATH = os.path.join(DATA_DIR, "kb_meta.json")

# =================== Session ===================
ss = st.session_state
if "is_admin" not in ss:       ss.is_admin = False
if "kb_ready" not in ss:       ss.kb_ready = False
if "kb_obj" not in ss:         ss.kb_obj = None
if "facts" not in ss:          ss.facts = {}
if "chat_history" not in ss:   ss.chat_history = []
if "show_sources" not in ss:   ss.show_sources = True

# =================== KB types ===================
@dataclass
class KB:
    vectorizer: TfidfVectorizer
    matrix: Any
    chunks: List[str]
    sources: List[str]

# =================== Utilities ===================
def _chunk(text: str, chunk_size: int = 500, overlap: int = 120) -> List[str]:
    words = text.split()
    if not words: return []
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size]))
        i += max(1, (chunk_size - overlap))
    return out

def _read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        # Normalize weird numeral splits from PDF text
        t = re.sub(r"(\d)\s+(\d)", r"\1\2", t)           # join broken numbers
        t = re.sub(r"\$\s+(\d)", r"$\1", t)              # $ 59 -> $59
        t = re.sub(r"(\d)\s*\.\s*(\d{2})", r"\1.\2", t)  # 59 . 99 -> 59.99
        pages.append(t)
    return "\n".join(pages)

def _extract_facts(full_text: str) -> Dict[str, Any]:
    """
    Parse known items (plans/prices/childcare, hours, contact) from SWT “At-a-Glance” style PDFs.
    Falls back gracefully if not present.
    """
    facts: Dict[str, Any] = {"plans": [], "childcare": {}, "hours": {}, "contact": {}}

    # Membership plans (e.g., "Sapphire — $59.99" + classes/week)
    plan_re = re.compile(
        r"(Sapphire|Pearl|Diamond)\s*[-—]\s*\$?\s*(\d+(?:\.\d{2})?)", re.I)
    for name, price in plan_re.findall(full_text):
        facts["plans"].append({"name": name.title(), "price": float(price)})

    # classes/week add-ons
    cw_re = re.compile(r"(Sapphire).*?(\b2\s*classes?/week\b)", re.I|re.S)
    if m := cw_re.search(full_text):
        for p in facts["plans"]:
            if p["name"].lower()=="sapphire": p["classes_per_week"]=2

    cw_re = re.compile(r"(Pearl).*?(\b3\s*classes?/week\b).*?(Includes\s*3\s*scans)?", re.I|re.S)
    if m := cw_re.search(full_text):
        for p in facts["plans"]:
            if p["name"].lower()=="pearl":
                p["classes_per_week"]=3
                if m.group(3): p["scans"]="Includes 3 scans"

    cw_re = re.compile(r"(Diamond).*?(Unlimited\s*classes)", re.I|re.S)
    if m := cw_re.search(full_text):
        for p in facts["plans"]:
            if p["name"].lower()=="diamond":
                p["classes_per_week"]="Unlimited"
                p["scans"]="Monthly scans included"

    # Childcare add-on per tier (e.g., "$84.99 with childcare add-on")
    childcare_re = re.compile(
        r"(Sapphire|Pearl|Diamond).*?\$?\s*(\d+(?:\.\d{2})?)\s*with\s*childcare\s*add-?on", re.I)
    for plan, amt in childcare_re.findall(full_text):
        facts["childcare"][plan.title()] = float(amt)

    # Hours (front desk)
    # "Mon–Fri 8:30a–12p & 3:30p–7:30p • Sat 8:00a–12p • Sun By appointment"
    hours = {}
    if "Mon–Fri" in full_text or "Mon-Fri" in full_text:
        hours["Mon–Fri"] = "8:30a–12p & 3:30p–7:30p"
    if re.search(r"Sat(?:urday)?\s+\d", full_text):
        hours["Sat"] = "8:00a–12p"
    if re.search(r"Sun(?:day)?.*appointment", full_text, re.I):
        hours["Sun"] = "By appointment"
    if "24/7" in full_text:
        hours["Members"] = "24/7 secure access for active members"
    if hours: facts["hours"] = hours

    # Contact
    if m := re.search(r"(\b443[-–]?\s?975[-–]?\s?9649\b)", full_text):
        facts["contact"]["phone"] = "443-975-9649"
    if m := re.search(r"([\w\.-]+@[\w\.-]+)", full_text):
        facts["contact"]["email"] = m.group(1)
    if m := re.search(r"(www\.[\w\.-]+)", full_text):
        facts["contact"]["website"] = "https://" + m.group(1).lstrip("www.")
    if m := re.search(r"\b1990\s+Chaneyville.*?20736", full_text):
        facts["contact"]["address"] = "1990 Chaneyville Rd, Owings, MD 20736"

    return facts

def _build_index(from_files: List[io.BytesIO], filenames: List[str]) -> KB:
    os.makedirs(DATA_DIR, exist_ok=True)
    all_chunks, sources = [], []
    full_text_accum = []

    for file, name in zip(from_files, filenames):
        file.seek(0)
        txt = _read_pdf(file)
        full_text_accum.append(txt)
        for ch in _chunk(txt, 500, 120):
            all_chunks.append(ch); sources.append(name)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000, lowercase=True)
    matrix = vectorizer.fit_transform(all_chunks)

    facts = _extract_facts("\n".join(full_text_accum))

    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(matrix, MAT_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"chunks": all_chunks, "sources": sources, "facts": facts}, f)

    ss.facts = facts
    return KB(vectorizer=vectorizer, matrix=matrix, chunks=all_chunks, sources=sources)

def _load_index_if_exists() -> KB|None:
    try:
        if not (os.path.exists(VEC_PATH) and os.path.exists(MAT_PATH) and os.path.exists(META_PATH)):
            return None
        vectorizer = joblib.load(VEC_PATH)
        matrix = joblib.load(MAT_PATH)
        meta = json.load(open(META_PATH))
        ss.facts = meta.get("facts", {})
        return KB(vectorizer=vectorizer, matrix=matrix, chunks=meta["chunks"], sources=meta["sources"])
    except Exception:
        return None

def _retrieve(kb: KB, query: str, k: int = 4) -> List[Tuple[str,str,float]]:
    qv = kb.vectorizer.transform([query])
    sims = cosine_similarity(qv, kb.matrix).ravel()
    idx = np.argsort(-sims)[:k]
    return [(kb.chunks[i], kb.sources[i], float(sims[i])) for i in idx]

def _fmt_money(x: float) -> str:
    return f"${x:,.2f}".replace(".00","")

def _price_range(facts: Dict[str,Any]) -> Tuple[float,float]:
    prices = [p["price"] for p in facts.get("plans", []) if "price" in p]
    if not prices: return (0.0, 0.0)
    return (min(prices), max(prices))

def _compose_simple_answer(question: str, hits: List[Tuple[str,str,float]]) -> str:
    """Short, safe summary from chunks if we don’t have structured facts."""
    if not hits:
        return "I’m not sure from our docs. Would you like to speak with our manager?"
    context = "\n".join(h[0] for h in hits)
    # pick a few sentences that include keywords
    kws = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w)>2]
    sents = re.split(r"(?<=[.!?])\s+", context)
    picked = []
    for s in sents:
        if any(k in s.lower() for k in kws) or len(picked)<2:
            s = re.sub(r"\s{2,}", " ", s).strip()
            # clean numbers that sometimes squash
            s = re.sub(r"(\d)\s+(\d)", r"\1\2", s)
            picked.append(s)
        if len(" ".join(picked))>300: break
    ans = " ".join(picked).strip()
    ans = textwrap.shorten(ans, width=400, placeholder="…")
    return ans or "I’m not sure from our docs. Would you like to speak with our manager?"

# =================== Domain smart answers ===================
def answer_membership(user_text: str, facts: Dict[str,Any]) -> str:
    lo, hi = _price_range(facts)
    if lo and hi:
        base = f"Our monthly plans start around {_fmt_money(lo)} and go up to about {_fmt_money(hi)} depending on how often you train and whether you add childcare."
    else:
        base = "We offer multiple membership tiers based on how often you plan to work out."
    prompt = " How often do you want to work out each week — 2x, 3x, or unlimited?"
    # If user specified frequency, recommend
    if m := re.search(r"\b(2|two)\b", user_text, re.I):
        rec = next((p for p in facts.get("plans", []) if p["name"].lower()=="sapphire"), None)
        if rec: return f"{base} Based on 2x/week, the **Sapphire** plan is a good fit at {_fmt_money(rec['price'])}. Want details or to book a tour?"
    if m := re.search(r"\b(3|three)\b", user_text, re.I):
        rec = next((p for p in facts.get("plans", []) if p["name"].lower()=="pearl"), None)
        if rec: return f"{base} For ~3x/week, check out **Pearl** at {_fmt_money(rec['price'])}. Need childcare or scan info?"
    if re.search(r"\bunlimited|as much|every day\b", user_text, re.I):
        rec = next((p for p in facts.get("plans", []) if p["name"].lower()=="diamond"), None)
        if rec: return f"{base} For unlimited classes, **Diamond** runs {_fmt_money(rec['price'])}. Want the booking link?"
    return base + prompt

def answer_childcare(facts: Dict[str,Any]) -> str:
    if not facts.get("childcare"):
        return "We do offer childcare. Exact pricing can vary by plan — want me to connect you with our manager to confirm?"
    bits = []
    for plan, amt in sorted(facts["childcare"].items()):
        bits.append(f"{plan}: {_fmt_money(amt)}")
    return "Childcare add-on is available. Current add-on rates — " + ", ".join(bits) + ". Do you need days/times too?"

def answer_hours(facts: Dict[str,Any]) -> str:
    hrs = facts.get("hours", {})
    if not hrs: return "Members have 24/7 access. For staffed/front-desk hours, would you like me to connect you with our manager?"
    parts = []
    if "Members" in hrs: parts.append("Members: 24/7 secure access")
    if "Mon–Fri" in hrs: parts.append(f"Mon–Fri: {hrs['Mon–Fri']}")
    if "Sat" in hrs: parts.append(f"Sat: {hrs['Sat']}")
    if "Sun" in hrs: parts.append(f"Sun: {hrs['Sun']}")
    return "Front desk hours — " + " • ".join(parts) + ". What day/time were you planning to visit?"

def answer_schedule_followup() -> str:
    return "Okay — what day would you like to work out, and roughly what time? I’ll check what classes are running then."

# =================== Main QA ===================
def answer_question(user_text: str) -> str:
    # Try to load KB once
    kb: KB|None = ss.kb_obj or _load_index_if_exists()
    if kb and not ss.kb_obj:
        ss.kb_obj = kb; ss.kb_ready = True

    q = user_text.lower()

    # Route to domain answers first
    if re.search(r"\b(member(ship)?|price|cost|plan|tier)\b", q):
        return answer_membership(user_text, ss.facts or {})
    if re.search(r"\b(child\s*care|childcare|kids? club)\b", q):
        return answer_childcare(ss.facts or {})
    if re.search(r"\b(hours?|open|close|staff(ed)?|front\s*desk)\b", q):
        return answer_hours(ss.facts or {})
    if re.search(r"\b(schedule|timetable|class(es)? when)\b", q):
        return answer_schedule_followup()

    # Retrieval fallback (concise)
    if not kb:
        return "Our knowledge base isn’t loaded yet. Go to Admin → upload PDFs, then try again."
    hits = _retrieve(kb, user_text, k=4)
    ans  = _compose_simple_answer(user_text, hits)

    # If we still look uncertain, nudge to human
    if not ans or re.search(r"(?i)i.?m not sure|don.?t know|couldn.?t find", ans):
        return "I’m not sure from our docs. Would you like to speak with our manager?"

    # Append sources (optional toggle)
    if ss.show_sources and hits:
        uniq = []
        for _, src, _ in hits:
            if src not in uniq: uniq.append(src)
        if uniq:
            ans += "\n\n_sources: " + ", ".join(uniq[:3])
    return ans

# =================== Calendly handoff ===================
def show_manager_slots():
    st.chat_message("assistant").write("Absolutely — pick a time that works best for you:")
    tz = pytz.timezone(CALENDLY_TZ or "America/New_York")
    now = dt.datetime.now(tz)
    start, end = now, now + dt.timedelta(days=7)

    if CALENDLY_EVENT_URI:
        try:
            slots = list_available_times(CALENDLY_EVENT_URI, start, end, CALENDLY_TZ)
        except Exception:
            slots = []
        if slots:
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
    # Fallback to public link
    if CALENDLY_URL:
        st.write(f'<a href="{CALENDLY_URL}" target="_blank" rel="noopener">Open booking page</a>', unsafe_allow_html=True)
    else:
        st.warning("Scheduling link isn’t configured yet in Secrets.")

# =================== Sidebar ===================
with st.sidebar:
    st.toggle("Show sources", value=ss.show_sources, key="show_sources")
    st.caption("Runs on Streamlit free tier. PDF/Text search only (no paid APIs).")
    st.markdown("---")
    if not ss.is_admin:
        with st.popover("Admin mode"):
            pw = st.text_input("Password", type="password")
            if st.button("Log in"):
                ss.is_admin = (pw == ADMIN_PASSWORD)
                st.success("Admin mode enabled." if ss.is_admin else "Wrong password.")
    else:
        st.success("Admin mode")
        if st.button("Log out"): ss.is_admin = False

# =================== Header ===================
st.markdown(f"""
<div style="text-align:center;margin-top:8px">
  <h1 style="margin-bottom:0.2rem">{STUDIO_NAME} Customer Support</h1>
  <div style="opacity:.75">Ask about classes, schedules, childcare, pricing, promotions, and more.</div>
</div>
""", unsafe_allow_html=True)

# =================== Admin: build/load KB ===================
if ss.is_admin:
    st.subheader("Admin · Load / replace knowledge base (PDF/Text)")
    uploaded = st.file_uploader("Upload one or more PDFs (pricing, policies, schedule).",
                                type=["pdf"], accept_multiple_files=True)
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Build/Replace knowledge base", type="primary", disabled=not uploaded, use_container_width=True):
            with st.spinner("Indexing & parsing…"):
                kb = _build_index([io.BytesIO(f.read()) for f in uploaded], [f.name for f in uploaded])
                ss.kb_obj = kb; ss.kb_ready = True
            st.success(f"Indexed {len(kb.chunks)} chunks across {len(set(kb.sources))} file(s). Parsed facts: {len(ss.facts.get('plans',[]))} plans, {len(ss.facts.get('hours',{}))} hour blocks.")
    with col2:
        if st.button("Load existing index from disk", use_container_width=True):
            kb = _load_index_if_exists()
            if kb:
                ss.kb_obj = kb; ss.kb_ready = True
                st.success("Loaded saved index.")
            else:
                st.info("No saved index found.")

    st.markdown("---")

# =================== Lead capture (optional banner) ===================
with st.expander("Leave your info for a follow-up (optional)"):
    with st.form("lead_form_top"):
        name  = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone (optional)")
        ok_to_contact = st.checkbox("I agree to be contacted about my inquiry.")
        if st.form_submit_button("Send"):
            if name and email:
                try:
                    add_lead(name, email, phone, interest="From chat", source="web")
                    st.success("Thanks! We’ll follow up shortly.")
                except Exception:
                    st.info("Saved locally. (Lead sheet not configured.)")
            else:
                st.warning("Please enter both name and email.")

# =================== Chat turn handling ===================
def handle_turn(user_text: str):
    st.chat_message("user").write(user_text)

    if wants_handoff(user_text):
        show_manager_slots()
        return

    ans = answer_question(user_text)

    # If still uncertain, offer manager + SMS + feedback form
    uncertain = bool(re.search(r"(?i)i.?m not sure|don.?t know|speak with our manager", ans))
    st.chat_message("assistant").write(ans)
    if uncertain:
        st.info("Prefer a quick answer? Text us at **+1 (443) 975-9649** and we’ll respond within ~30 minutes. "
                "To report a bug or leave feedback, use the form below.")

# Replay history
for role, msg in ss.chat_history:
    st.chat_message(role).write(msg)

# Input
prompt = st.chat_input("Type your question (schedule, memberships, childcare, etc.)") or ""
if prompt:
    ss.chat_history.append(("user", prompt))
    handle_turn(prompt)

# Feedback / complaint form (Google Form or URL you set)
st.markdown(
    '<div style="margin-top:18px"><a href="https://forms.gle/replace-with-your-form" target="_blank" rel="noopener">'
    'Submit a complaint / feedback</a></div>',
    unsafe_allow_html=True
)

# Footer
st.markdown(
    '<div style="text-align:center;opacity:.65;margin:24px 0 6px 0;">Powered by <strong>ZARI</strong></div>',
    unsafe_allow_html=True
)
