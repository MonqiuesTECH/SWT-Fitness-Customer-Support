# leads.py — safe when Google secrets are missing
import json, datetime as dt
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def _get_ws():
    try:
        sa_json = st.secrets.get("GOOGLE_SA_JSON")
        sheet_id = st.secrets.get("LEADS_SHEET_ID")
        if not (sa_json and sheet_id and gspread and Credentials):
            return None
        creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=SCOPES)
        gc = gspread.authorize(creds)
        return gc.open_by_key(sheet_id).worksheet("Leads")
    except Exception:
        return None

def add_lead(name: str, email: str, phone: str, interest: str, source: str = "web") -> bool:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    ws = _get_ws()
    if ws is None:
        st.session_state.setdefault("leads_local", []).append([ts, name, email, phone, interest, source])
        return False
    ws.append_row([ts, name, email, phone, interest, source], value_input_option="USER_ENTERED")
    return True
