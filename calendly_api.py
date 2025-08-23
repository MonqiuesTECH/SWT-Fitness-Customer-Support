# calendly_api.py
"""
Calendly minimal API client for Streamlit.
Reads PAT from st.secrets when not provided explicitly.
"""

import datetime as dt
from typing import List, Dict, Any, Optional

import requests
import streamlit as st

API = "https://api.calendly.com"

def _headers(pat: Optional[str] = None) -> Dict[str, str]:
    token = pat or st.secrets.get("CALENDLY_PAT", "")
    if not token:
        raise RuntimeError("CALENDLY_PAT missing in secrets.")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

def list_available_times(
    event_type_uri: str,
    start: dt.datetime,
    end: dt.datetime,
    tz: str,
    pat: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of time slots with their scheduling URLs for a 7-day window.
    """
    params = {
        "event_type": event_type_uri,
        "start_time": start.replace(microsecond=0).isoformat(),
        "end_time": end.replace(microsecond=0).isoformat(),
        "timezone": tz,
    }
    r = requests.get(f"{API}/event_type_available_times", headers=_headers(pat), params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("collection", [])

def create_single_use_link(event_type_uri: str, max_events: int = 1, pat: Optional[str] = None) -> str:
    """
    Generates a single-use booking URL scoped to the event type.
    """
    payload = {"max_event_count": max_events, "owner": event_type_uri, "owner_type": "EventType"}
    r = requests.post(f"{API}/scheduling_links", headers={**_headers(pat), "Content-Type": "application/json"}, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()["resource"]["booking_url"]
