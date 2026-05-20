# src/tools/web_search.py
"""
Tool 3: Real-Time Web Search
Uses DuckDuckGo to search the internet for current information.
Free, no API key needed.
Use this for: current events, recent news, live information.
"""

from typing import Optional
from langchain_core.tools import Tool
from ddgs import DDGS


def search_web(query: str, max_results: int = 4) -> str:
    try:
        ddgs = DDGS()
        search_results = list(
            ddgs.text(query, max_results=max_results)
        )

        if not search_results:
            return f"No web results found for: {query}"

        results = []
        for i, result in enumerate(search_results, start=1):
            title   = result.get("title",   "No title")
            url     = result.get("href",    "No URL")
            snippet = result.get("body",    "No description")
            results.append(
                f"[Result {i}]\n"
                f"Title   : {title}\n"
                f"URL     : {url}\n"
                f"Summary : {snippet}\n"
            )
        return "\n---\n".join(results)

    except Exception as e:
        return f"Web search failed: {str(e)}"


def search_news(query: str, max_results: int = 4) -> str:
    try:
        ddgs = DDGS()
        news_results = list(
            ddgs.news(query, max_results=max_results)
        )

        if not news_results:
            return f"No news found for: {query}"

        results = []
        for i, article in enumerate(news_results, start=1):
            title  = article.get("title",  "No title")
            url    = article.get("url",    "No URL")
            body   = article.get("body",   "No content")
            date   = article.get("date",   "Unknown date")
            source = article.get("source", "Unknown source")
            results.append(
                f"[Article {i}]\n"
                f"Title   : {title}\n"
                f"Source  : {source} | Date: {date}\n"
                f"URL     : {url}\n"
                f"Summary : {body}\n"
            )
        return "\n---\n".join(results)

    except Exception as e:
        return f"News search failed: {str(e)}"

def smart_web_search(query: str) -> str:
    """
    Automatically decide between general search and news search.
    News keywords trigger news search, everything else is general.
    """
    news_keywords = [
        "news", "today", "latest", "recent", "current",
        "happened", "update", "announced", "2024", "2025"
    ]

    is_news = any(
        kw in query.lower() for kw in news_keywords
    )

    if is_news:
        print(f" Using NEWS search for: {query}")
        return search_news(query)
    else:
        print(f" Using WEB search for: {query}")
        return search_web(query)


# ── Wrap as LangChain Tool ────────────────────────────────────
web_search_tool = Tool(
    name="web_search",
    func=smart_web_search,
    description=(
        "Search the internet for real-time, current information. "
        "Use this tool when the user asks about recent events, "
        "current news, latest updates, today's information, or "
        "anything that requires up-to-date knowledge beyond the "
        "local knowledge base. Input should be a search query string."
    )
)


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- General Search ---")
    result = search_web("LangChain framework features 2024")
    print(result[:500])

    print("\n--- News Search ---")
    result = search_news("AI news today")
    print(result[:500])