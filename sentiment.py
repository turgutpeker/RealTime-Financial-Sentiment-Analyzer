import streamlit as st
from transformers import pipeline

LABEL_MAP = {
    "positive": "Bullish",
    "negative": "Bearish",
    "neutral": "Neutral",
}

EMOJI_MAP = {
    "Bullish": "🟢",
    "Bearish": "🔴",
    "Neutral": "🟡",
}

COLOR_MAP = {
    "Bullish": "#2ecc71",
    "Bearish": "#e74c3c",
    "Neutral": "#f39c12",
}


@st.cache_resource(show_spinner="Loading FinBERT model…")
def _load_model():
    """Load FinBERT once and cache it for the lifetime of the Streamlit server."""
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
    )


def analyze_sentiment(text: str) -> dict:
    """Return sentiment label, emoji, color, and confidence score for a single text."""
    classifier = _load_model()
    result = classifier(text, truncation=True, max_length=512)[0]
    label = LABEL_MAP[result["label"].lower()]
    return {
        "sentiment": label,
        "emoji": EMOJI_MAP[label],
        "color": COLOR_MAP[label],
        "score": round(result["score"], 4),
    }


def analyze_batch(texts: list[str]) -> list[dict]:
    """Analyze a list of texts efficiently using batched inference."""
    classifier = _load_model()
    raw_results = classifier(texts, truncation=True, max_length=512, batch_size=8)
    output = []
    for result in raw_results:
        label = LABEL_MAP[result["label"].lower()]
        output.append({
            "sentiment": label,
            "emoji": EMOJI_MAP[label],
            "color": COLOR_MAP[label],
            "score": round(result["score"], 4),
        })
    return output
