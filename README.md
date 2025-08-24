SWT Fitness – Customer Support Bot (Streamlit)

https://www.loom.com/share/9b29a521131946c1b2683b7773eda4fd?sid=0a679102-61cb-4069-abaa-4353d0987e6a

A lightweight, free-tier friendly customer support assistant for a women’s gym.
Built with Streamlit, TF-IDF RAG (no paid LLM APIs), Calendly handoff, and Google Sheets lead capture.

Concise answers grounded in your PDFs (pricing, schedule, policy).

Dialog flows to avoid confusion:

Memberships: gives a price range, asks “how often?”, then recommends a plan.

Schedule: asks day → time → shows classes that match.

Human handoff: “Speak to a person” opens your Calendly booking page (or API time slots if enabled).

Leads: welcome gate + “buying intent” capture → logs to a Google Sheet.

https:// (your Streamlit Cloud URL)

Contents

Demo

Features

Architecture

Getting Started

Secrets Configuration

Local Development

Deploy to Streamlit Cloud

Embedding on Wix

Repository Structure

Troubleshooting

Security Notes

Roadmap

License

Demo

User: “how much is membership?”
Bot: “Our memberships range from $X–$Y. How often do you want to go each week? (2 / 3 / unlimited)”

User: “3 times”
Bot: “I’d recommend Pearl: $79.99 (3/week); $104.99 with childcare. Want me to help you book a quick call to get started?”

User: “what’s the schedule?”
Bot: “Okay — what day would you like to work out?” → “Great — what time works? (morning / afternoon / evening or a time like 6:00am)” → lists classes.

If the bot isn’t sure:
Bot: “I’m not sure — would you like to speak with the manager?” (shows Calendly link/slots)

Features

RAG from PDFs: uploads membership pricing, schedules, policies; indexes with scikit-learn TF-IDF; no paid LLM calls.

Concise output: never dumps raw PDF paragraphs; short answers only.

Dialog manager: explicit slot-filling for membership and schedule queries.

Handoff: one-click booking via Calendly.

Lead capture: welcome gate + “buying intent” form → Google Sheets.

Architecture

Frontend/Runtime: Streamlit (Free tier OK)

Indexing: pypdf extract → chunk → TfidfVectorizer → cosine similarity

Storage: indexes saved to a writable temp dir (/tmp on Streamlit Cloud).

Dialogs:

Membership: parse tier names/prices from PDFs → compute range → ask frequency → recommend tier.

Schedule: build a simple day/time index from PDFs → ask day/time → display matches.

Handoff: Calendly public URL (or API event-type URI if you choose).

Leads: Google Service Account → gspread → append rows to your Sheet.

Getting Started

Clone

git clone https://github.com/<you>/<repo>.git
cd <repo>


Install

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt


Secrets

Create .streamlit/secrets.toml (see next section).

Do not commit secrets.

Run

streamlit run app.py


Admin → Upload PDFs

Click the Admin popover (password from secrets).

Upload your PDFs (Pricing, Schedule, Policy, etc.) → Build/Replace knowledge base.

Secrets Configuration

Create .streamlit/secrets.toml:

# App
STUDIO_NAME = "SWT Fitness"
ADMIN_PASSWORD = "SWT_2025!manager"
UNKNOWN_MESSAGE = "I'm not sure — would you like to speak with the manager?"
MIN_SIMILARITY = 0.12
CALENDLY_TZ = "America/New_York"

# Calendly (use public link to avoid API complexity)
CALENDLY_EVENT_TYPE = ""  # leave blank to disable API slots
CALENDLY_URL = "https://calendly.com/swtfitness2020/swt-meeting-w-sandy"

# Google Sheets (Leads)
LEADS_SHEET_ID   = "<your-spreadsheet-id>"
LEADS_WORKSHEET  = "SWT AI Bot Leads"
GOOGLE_SERVICE_ACCOUNT = """{ ...full JSON key... }"""


Share the Sheet with your service account email (Editor).
Service account email looks like: ...@<project>.iam.gserviceaccount.com.

Optional:

# Persist index somewhere else (defaults to /tmp on Streamlit Cloud)
DATA_DIR = "/app/data"

Local Development

Python: 3.10–3.12 recommended

requirements.txt

streamlit>=1.36
pypdf>=6.0.0
scikit-learn==1.7.1
numpy>=2.3.0
scipy==1.16.1
joblib>=1.3.2
pytz
requests
gspread
google-auth

Deploy to Streamlit Cloud

Push your repo to GitHub.

Streamlit Cloud → New app → select this repo/branch.

App Settings → Secrets → paste the same content as .streamlit/secrets.toml.

Deploy.

On free tier, the app writes to /tmp (already handled). No extra config needed.

Embedding on Wix

Wix Editor → Add (+) → Embed → Embed a Site → URL:

https://<your-app>.streamlit.app/?embed=true


Set width 100%, height 900–1100px, scrolling on.

Or use a custom HTML block:

<iframe
  src="https://<your-app>.streamlit.app/?embed=true"
  width="100%" height="1000" style="border:0"
  allow="clipboard-write" loading="lazy">
</iframe>

Repository Structure
.
├── app.py                    # Streamlit app (dialogs, KB, UI)
├── intents.py                # detects “speak with someone” and related phrases
├── calendly_api.py           # (optional) API slot buttons; public link fallback supported
├── leads.py                  # Google Sheets integration (service account)
├── requirements.txt
├── README.md
├── LICENSE
└── .streamlit/
    └── secrets.toml          # NOT COMMITTED – add in Streamlit Cloud settings

Troubleshooting

Opens wrong scheduling page / shows red error

Set CALENDLY_EVENT_TYPE = "" and only keep CALENDLY_URL with your public booking link. Save Secrets → Rerun.

PDF upload error: permission denied to /mnt/data

This code writes to /tmp by default (Streamlit free-tier safe). If you changed DATA_DIR, ensure it’s writable.

Leads not writing to Sheets

Share the sheet with the service account (Editor).

Verify LEADS_SHEET_ID and LEADS_WORKSHEET names match.

Answers too long / off-topic

The app never pastes raw paragraphs. If docs are messy, keep pricing/schedule pages text-based (not scanned).

You can tighten MIN_SIMILARITY in secrets (e.g., 0.15).

Bot uncertainty

If it can’t ground an answer, it will always say:
“I’m not sure — would you like to speak with the manager?” and show Calendly.

Security Notes

Never commit .streamlit/secrets.toml or service-account JSON to Git.

Rotate tokens/keys periodically (Calendly PAT, Google key).

Streamlit Secrets are encrypted at rest in Streamlit Cloud.

Roadmap

Better schedule parsing (structured CSV/Sheet → richer time filters).

Optional Twilio SMS for automated texting flows (not free; swap in later).

Admin dashboard: lead stats, top FAQs, content freshness checks.

Multi-doc versioning (effective dates) + conflict detection.
