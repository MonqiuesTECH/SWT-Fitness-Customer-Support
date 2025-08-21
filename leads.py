# leads.py
import json, datetime as dt
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SA = json.loads(st.secrets["GOOGLE_SA_JSON"])
CREDS = Credentials.from_service_account_info(SA, scopes=SCOPES)
GC = gspread.authorize(CREDS)
SHEET = GC.open_by_key(st.secrets["LEADS_SHEET_ID"]).worksheet("Leads")

def add_lead(name: str, email: str, phone: str, interest: str, source: str = "chat"):
    ts = dt.datetime.now().isoformat(timespec="seconds")
    SHEET.append_row([ts, name, email, phone, interest, source], value_input_option="USER_ENTERED")
