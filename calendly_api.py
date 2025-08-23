# calendly_api.py
# Minimal Calendly client with UTC time formatting + diagnostics helpers.

from __future__ import annotations
import datetime as dt
from typing import Dict, Any, List, Optional

import requests
import streamlit as st

API = "https://api.calendly.com"

def _pat(pat: Optional[str] = None) -> str:
    tok = pat or st.secrets.get("CALENDLY_PAT", "")
    if not tok:
        raise RuntimeError("CALENDLY_PAT missing in secrets.")
    return tok

def _headers(pat: Optional[str] = None) -> Dict[str, str]:
    return {"Authorization": f"Bearer {_pat(pat)}", "Accept": "application/json"}

def _to_utc_z(t: dt.datetime) -> str:
    # Calendly expects ISO8601 in UTC; pass timezone separately via "timezone" param
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    t_utc = t.astimezone(dt.timezone.utc)
    return t_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")

# --- Public helpers ---

def whoami(pat: Optional[str] = None) -> Dict[str, Any]:
    r = requests.get(f"{API}/users/me", headers=_headers(pat), timeout=20)
    r.raise_for_status()
    return r.json()["resource"]  # has 'uri', 'current_organization', etc.

def list_event_types(user_uri: str, pat: Optional[str] = None) -> List[Dict[str, Any]]:
    r = requests.get(f"{API}/event_types", headers=_headers(pat), params={"user": user_uri}, timeout=20)
    r.raise_for_status()
    return r.json().get("collection", [])

def list_available_times(event_type_uri: str, start: dt.datetime, end: dt.datetime, tz: str,
                         pat: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {
        "event_type": event_type_uri,
        "start_time": _to_utc_z(start),
        "end_time": _to_utc_z(end),
        "timezone": tz,
    }
    r = requests.get(f"{API}/event_type_available_times", headers=_headers(pat), params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("collection", [])

def create_single_use_link(event_type_uri: str, max_events: int = 1, pat: Optional[str] = None) -> str:
    payload = {"max_event_count": max_events, "owner": event_type_uri, "owner_type": "EventType"}
    r = requests.post(f"{API}/scheduling_links", headers={**_headers(pat), "Content-Type": "application/json"},
                      json=payload, timeout=20)
    r.raise_for_status()
    return r.json()["resource"]["booking_url"]
