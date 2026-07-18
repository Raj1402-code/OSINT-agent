# OSINT Investigative Agent

A Streamlit app that investigates a broad question by planning sub-questions,
searching the web (DuckDuckGo, no API key needed), scraping sources,
extracting atomic evidence, cross-checking for conflicts, and producing a
final cited answer — powered by Google's **free** Gemini API.

## 1. Prerequisites

- Python 3.10+ installed
- A free Gemini API key: https://aistudio.google.com/apikey (sign in with a
  Google account, click "Create API key" — no credit card required)

## 2. Setup (run these in Terminal, inside the unzipped `osint-agent` folder)

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
```

Now open `.env` in a text editor and replace `your-gemini-api-key-here` with
your real Gemini API key.

## 3. Run it

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

## 4. Using the app

Type an investigative question into the chat box, e.g.:

> "What is the current status of [some public policy debate]?"

The agent will show its progress (planning, searching, reading sources) in a
live status box, then produce:
- A cited answer (inline `[S1]`, `[S2]`... references)
- A confidence level (High / Medium / Low) with rationale
- Any conflicting claims found across sources
- A full source list with the specific claims drawn from each one

## 5. Project structure

```
osint-agent/
├── app.py                  # Streamlit UI + pipeline orchestration
├── agent/
│   ├── utils.py             # Settings, logging, text/URL helpers
│   ├── search.py            # DuckDuckGo web search
│   ├── planner.py           # Gemini: question -> sub-questions
│   ├── extractor.py         # Page scraping + Gemini: page -> evidence
│   ├── verifier.py          # Gemini: evidence -> conflicts + confidence
│   └── answer.py            # Gemini: evidence -> final cited answer
├── requirements.txt
├── .env.example
└── .gitignore
```

## 6. Notes

- Gemini's free tier has its own per-minute rate limits. If you run many
  sub-questions back to back you may occasionally see a `429` error in the
  logs — the app catches this gracefully and degrades (e.g. lower confidence)
  rather than crashing.
- No content is ever fabricated: every extracted claim must trace back to a
  supporting quote from an actual scraped page.
- To deploy this publicly later (optional), push this folder to a GitHub
  repo and connect it via Streamlit Community Cloud, adding `GEMINI_API_KEY`
  as a secret in the app's settings (never commit your real `.env` file).
