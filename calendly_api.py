# calendly_api.py
import datetime as dt
import requests

API = "https://api.calendly.com"

def list_available_times(token: str, event_type_uri: str, start: dt.datetime, end: dt.datetime, tz: str):
    """Return Calendly available times for a given event type and window (max 7 days)."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {
        "event_type": event_type_uri,
        "start_time": start.replace(microsecond=0).isoformat(),
        "end_time": end.replace(microsecond=0).isoformat(),
        "timezone": tz,
    }
    r = requests.get(f"{API}/event_type_available_times", headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("collection", [])

def create_single_use_link(token: str, event_type_uri: str, max_events: int = 1):
    """Optional helper to generate a single-use scheduling link."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {"max_event_count": max_events, "owner": event_type_uri, "owner_type": "EventType"}
    r = requests.post(f"{API}/scheduling_links", headers=headers, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()["resource"]["booking_url"]
