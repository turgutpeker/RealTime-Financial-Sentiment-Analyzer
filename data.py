import requests

MOCK_COMMENTS = [
    "Tesla stock is going to the moon! Strong earnings beat expectations.",
    "Market looks terrible, recession is coming and inflation is out of control.",
    "I'm not sure about Apple earnings this quarter, could go either way.",
    "Huge breakout in crypto today! Bitcoin smashing through resistance levels.",
    "This stock is dead, sell everything before it collapses completely.",
    "Fed rate decision caused massive uncertainty across all sectors.",
    "Goldman Sachs upgraded the stock to buy — strong growth outlook ahead.",
    "Oil prices are crashing hard, energy sector in serious trouble.",
    "Solid quarterly revenue from Microsoft, cloud business is thriving.",
    "The housing market is showing serious cracks, mortgage rates are brutal.",
    "NVIDIA's AI chip demand is insane, supply can't keep up.",
    "Retail sales data came in weak — consumer spending is slowing down.",
    "Warren Buffett increased his position in this stock — bullish signal.",
    "Earnings miss again, management has no idea what they're doing.",
    "Dividend cut announced — major red flag for long-term investors.",
]

REDDIT_SUBREDDITS = ["stocks", "investing", "wallstreetbets"]


def get_mock_comments() -> list[str]:
    """Return the static mock comment list."""
    return MOCK_COMMENTS.copy()


def get_reddit_comments(subreddit: str = "stocks", limit: int = 25) -> list[str]:
    """
    Fetch top post titles from a subreddit using the public JSON API.
    Falls back to mock data if the request fails.
    """
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    headers = {"User-Agent": "FinancialSentimentAnalyzer/1.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        posts = [
            post["data"]["title"]
            for post in data["data"]["children"]
            if not post["data"].get("stickied", False)
        ]
        return posts[:limit]
    except Exception:
        return []


def get_comments(source: str = "mock", subreddit: str = "stocks") -> list[str]:
    """
    Unified entry point for fetching comments.

    Args:
        source:    "mock" for static data, "reddit" for live Reddit posts.
        subreddit: Which subreddit to scrape when source is "reddit".
    """
    if source == "reddit":
        comments = get_reddit_comments(subreddit)
        if comments:
            return comments
    return get_mock_comments()
