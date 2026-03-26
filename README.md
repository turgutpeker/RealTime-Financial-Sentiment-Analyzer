# 📊 Real-Time Financial Sentiment Analyzer

Classifies financial text as **🟢 Bullish**, **🔴 Bearish**, or **🟡 Neutral** using [FinBERT](https://huggingface.co/ProsusAI/finbert) — a finance-specific BERT model from HuggingFace. Runs entirely locally with no paid APIs or API keys.

**[Live Demo →](https://your-app-name.streamlit.app)** *(replace after deploying)*

---

## Features

- **Real-time sentiment classification** — Bullish / Bearish / Neutral labels with confidence scores
- **Financial NLP model** — FinBERT, trained specifically on financial text (not generic BERT)
- **Interactive dashboard** — KPI cards, pie chart, bar chart, per-comment confidence visualization
- **Live Feed simulation** — watch comments stream in and get classified one by one
- **Live Reddit data** — pull real post titles from r/stocks, r/investing, r/wallstreetbets
- **Custom text analyzer** — type any headline and get an instant prediction
- **100% free** — no OpenAI, no paid APIs, no API keys needed

---

## Tech Stack

| Layer | Tool |
|---|---|
| Sentiment Model | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) |
| ML Framework | HuggingFace Transformers + PyTorch |
| Dashboard | Streamlit |
| Charts | Matplotlib |
| Data | Mock data or Reddit public JSON API |
| Language | Python 3.10+ |

---

## Project Structure

```
financial-sentiment-analyzer/
├── app.py            # Streamlit dashboard (tabs: Dashboard / Live Feed / Analyze Text)
├── sentiment.py      # FinBERT model wrapper with @st.cache_resource
├── data.py           # Data sources (mock + Reddit scraper)
├── requirements.txt
└── README.md
```

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/financial-sentiment-analyzer.git
cd financial-sentiment-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

> **Note:** The first run downloads the FinBERT model (~440 MB). It is cached locally after that.

---

## Deploy to Streamlit Cloud (Free Public URL)

1. Push this repo to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Click **New App**
4. Select your repo, set branch to `main`, and set **Main file** to `app.py`
5. Click **Deploy**

You will get a public URL like `https://your-app-name.streamlit.app` — ready to share on your CV.

> Streamlit Cloud is completely free for public repos. The model is downloaded on first cold start (~2 min), then cached.

---

## How It Works

1. `data.py` fetches comments (mock or live Reddit).
2. `sentiment.py` loads FinBERT once via `@st.cache_resource` (shared across all sessions).
3. Comments are batched through the model; labels are mapped:
   - `positive` → 🟢 **Bullish**
   - `negative` → 🔴 **Bearish**
   - `neutral`  → 🟡 **Neutral**
4. `app.py` renders results across three tabs: Dashboard, Live Feed, and custom text input.

---

## Screenshots

| Dashboard | Live Feed |
|---|---|
| *KPI cards + pie/bar/confidence charts* | *Streaming comments with colored badges* |
