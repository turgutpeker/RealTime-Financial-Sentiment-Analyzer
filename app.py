import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data import get_comments, REDDIT_SUBREDDITS
from sentiment import analyze_batch, analyze_sentiment, COLOR_MAP

st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    data_source = st.radio(
        "Data source",
        options=["Mock Data", "Reddit (live)"],
        index=0,
    )

    subreddit = "stocks"
    if data_source == "Reddit (live)":
        subreddit = st.selectbox("Subreddit", REDDIT_SUBREDDITS)

    run_button = st.button("🔄 Analyze", use_container_width=True)

    st.markdown("---")
    st.caption("Model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)")
    st.caption("100% free · runs locally · no API keys needed")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📊 Real-Time Financial Sentiment Analyzer")
st.markdown(
    "Classify financial text as **🟢 Bullish**, **🔴 Bearish**, or **🟡 Neutral** "
    "using [FinBERT](https://huggingface.co/ProsusAI/finbert) — a finance-specific NLP model."
)

# ── Fetch & analyze data ──────────────────────────────────────────────────────
if run_button or "results_df" not in st.session_state:
    source_key = "reddit" if data_source == "Reddit (live)" else "mock"

    with st.spinner("Fetching data…"):
        comments = get_comments(source=source_key, subreddit=subreddit)

    if not comments:
        st.error("Could not fetch Reddit data. Check your connection or switch to Mock Data.")
        st.stop()

    with st.spinner(f"Analyzing {len(comments)} comments with FinBERT…"):
        results = analyze_batch(comments)

    rows = [
        {
            "Comment": comment,
            "Sentiment": f"{r['emoji']} {r['sentiment']}",
            "Confidence": r["score"],
            "_raw": r["sentiment"],
        }
        for comment, r in zip(comments, results)
    ]
    st.session_state["results_df"] = pd.DataFrame(rows)
    st.session_state["comments"] = comments
    st.session_state["results"] = results

df: pd.DataFrame = st.session_state["results_df"]
comments: list[str] = st.session_state["comments"]
results: list[dict] = st.session_state["results"]
raw_sentiments: list[str] = df["_raw"].tolist()

# ── KPI cards ─────────────────────────────────────────────────────────────────
counts = pd.Series(raw_sentiments).value_counts()
bullish = int(counts.get("Bullish", 0))
bearish = int(counts.get("Bearish", 0))
neutral = int(counts.get("Neutral", 0))
total = len(raw_sentiments)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Comments", total)
col2.metric("🟢 Bullish", bullish, f"{bullish/total*100:.0f}%")
col3.metric("🔴 Bearish", bearish, f"{bearish/total*100:.0f}%")
col4.metric("🟡 Neutral", neutral, f"{neutral/total*100:.0f}%")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_feed, tab_input = st.tabs(["📊 Dashboard", "⏱️ Live Feed", "🔍 Analyze Text"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════════════════════════════════════
with tab_dash:
    left, mid, right = st.columns(3)

    # Pie chart — sentiment distribution
    with left:
        st.subheader("Sentiment Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        pie_labels = [l for l, v in [("Bullish", bullish), ("Bearish", bearish), ("Neutral", neutral)] if v > 0]
        pie_values = [v for v in [bullish, bearish, neutral] if v > 0]
        pie_colors = [COLOR_MAP[l] for l in pie_labels]
        wedges, texts, autotexts = ax_pie.pie(
            pie_values,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.0f%%",
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for t in autotexts:
            t.set_fontsize(12)
            t.set_color("white")
            t.set_fontweight("bold")
        ax_pie.set_facecolor("none")
        fig_pie.patch.set_alpha(0)
        st.pyplot(fig_pie, use_container_width=True)

    # Bar chart — sentiment count
    with mid:
        st.subheader("Count by Label")
        fig_bar, ax_bar = plt.subplots(figsize=(4, 4))
        bar_labels = ["Bullish", "Bearish", "Neutral"]
        bar_values = [bullish, bearish, neutral]
        bar_colors = [COLOR_MAP[l] for l in bar_labels]
        bars = ax_bar.bar(bar_labels, bar_values, color=bar_colors, edgecolor="white", linewidth=1.5)
        ax_bar.set_ylabel("Number of comments")
        ax_bar.set_ylim(0, max(bar_values) + 2)
        for bar, val in zip(bars, bar_values):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(val),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax_bar.spines[["top", "right"]].set_visible(False)
        fig_bar.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)

    # Horizontal bar — confidence scores
    with right:
        st.subheader("Confidence Scores")
        fig_conf, ax_conf = plt.subplots(figsize=(4, 4))
        conf_labels = [c[:35] + "…" if len(c) > 35 else c for c in comments]
        conf_values = [r["score"] for r in results]
        conf_colors = [r["color"] for r in results]
        ax_conf.barh(conf_labels, conf_values, color=conf_colors, edgecolor="white")
        ax_conf.set_xlim(0, 1)
        ax_conf.set_xlabel("Confidence")
        ax_conf.invert_yaxis()
        ax_conf.spines[["top", "right"]].set_visible(False)
        ax_conf.tick_params(axis="y", labelsize=7)
        fig_conf.tight_layout()
        st.pyplot(fig_conf, use_container_width=True)

    st.markdown("---")

    # Filterable data table
    st.subheader("📄 Detailed Results")
    sentiment_filter = st.multiselect(
        "Filter by sentiment",
        options=["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"],
        default=["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"],
    )
    display_df = df.drop(columns=["_raw"])
    filtered_df = display_df[display_df["Sentiment"].isin(sentiment_filter)] if sentiment_filter else display_df
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Feed Simulation
# ════════════════════════════════════════════════════════════════════════════════
with tab_feed:
    st.subheader("⏱️ Live Feed Simulation")
    st.markdown(
        "Watch comments stream in and get classified in real time. "
        "Press **▶ Start Feed** to replay the animation."
    )

    speed = st.select_slider(
        "Feed speed",
        options=["Slow (1s)", "Normal (0.5s)", "Fast (0.2s)"],
        value="Normal (0.5s)",
    )
    delay_map = {"Slow (1s)": 1.0, "Normal (0.5s)": 0.5, "Fast (0.2s)": 0.2}
    delay = delay_map[speed]

    if st.button("▶ Start Feed", use_container_width=True):
        feed_placeholder = st.empty()
        feed_rows: list[dict] = []

        for comment, result in zip(comments, results):
            feed_rows.append(
                {
                    "emoji": result["emoji"],
                    "sentiment": result["sentiment"],
                    "comment": comment,
                    "score": result["score"],
                }
            )
            with feed_placeholder.container():
                for item in feed_rows:
                    badge_color = COLOR_MAP[item["sentiment"]]
                    st.markdown(
                        f"""
                        <div style="
                            border-left: 4px solid {badge_color};
                            padding: 8px 14px;
                            margin-bottom: 8px;
                            border-radius: 4px;
                            background: rgba(0,0,0,0.04);
                        ">
                            <span style="font-size:1.1em; font-weight:600;">{item['emoji']} {item['sentiment']}</span>
                            &nbsp;&nbsp;<span style="color:#888; font-size:0.8em;">{item['score']:.0%} confidence</span>
                            <br><span style="font-size:0.95em;">{item['comment']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            time.sleep(delay)

        st.success(f"Feed complete — {len(comments)} comments processed.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Custom Text Analyzer
# ════════════════════════════════════════════════════════════════════════════════
with tab_input:
    st.subheader("🔍 Analyze Your Own Text")
    st.markdown("Type any financial headline, comment, or tweet and get an instant prediction.")

    user_input = st.text_area(
        "Financial comment or headline:",
        placeholder="e.g. Apple reported record-breaking profits this quarter…",
        height=120,
    )

    if st.button("Analyze", use_container_width=True) and user_input.strip():
        with st.spinner("Analyzing…"):
            result = analyze_sentiment(user_input.strip())

        badge_color = result["color"]
        st.markdown(
            f"""
            <div style="
                border-left: 5px solid {badge_color};
                padding: 16px 20px;
                border-radius: 6px;
                background: rgba(0,0,0,0.04);
                font-size: 1.1em;
            ">
                <strong>Verdict:</strong> {result['emoji']} {result['sentiment']}<br>
                <strong>Confidence:</strong> {result['score']:.2%}
            </div>
            """,
            unsafe_allow_html=True,
        )
