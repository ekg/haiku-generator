"""
fetch_news module - Retrieves news from an external API source.

This module provides functionality to fetch news articles from a news API.
It supports fetching top headlines and searching for news by keyword.
"""

import json
import os
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional


# Default API configuration
NEWS_API_BASE_URL = os.environ.get("NEWS_API_BASE_URL", "https://newsapi.org/v2")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")


class NewsAPIError(Exception):
    """Custom exception for News API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class NewsArticle:
    """Represents a single news article."""

    def __init__(self, title: str, description: str, url: str, source: str,
                 published_at: str, author: Optional[str] = None,
                 image_url: Optional[str] = None, content: Optional[str] = None):
        self.title = title
        self.description = description
        self.url = url
        self.source = source
        self.published_at = published_at
        self.author = author
        self.image_url = image_url
        self.content = content

    def to_dict(self) -> dict:
        """Convert the article to a dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at,
            "author": self.author,
            "image_url": self.image_url,
            "content": self.content,
        }

    def __repr__(self) -> str:
        return f"NewsArticle(title='{self.title[:50]}...', source='{self.source}')"


class NewsResponse:
    """Represents the response from the News API."""

    def __init__(self, status: str, total_results: int, articles: list):
        self.status = status
        self.total_results = total_results
        self.articles = articles

    def to_dict(self) -> dict:
        """Convert the response to a dictionary."""
        return {
            "status": self.status,
            "total_results": self.total_results,
            "articles": [a.to_dict() for a in self.articles],
        }


def _parse_article(data: dict) -> NewsArticle:
    """Parse a raw article dict from the API into a NewsArticle object."""
    source = data.get("source", {})
    source_name = source.get("name", "Unknown") if isinstance(source, dict) else str(source)

    return NewsArticle(
        title=data.get("title", ""),
        description=data.get("description", ""),
        url=data.get("url", ""),
        source=source_name,
        published_at=data.get("publishedAt", ""),
        author=data.get("author"),
        image_url=data.get("urlToImage"),
        content=data.get("content"),
    )


def _make_request(endpoint: str, params: dict, api_key: Optional[str] = None) -> dict:
    """
    Make an HTTP GET request to the News API.

    Args:
        endpoint: The API endpoint (e.g., 'top-headlines', 'everything').
        params: Query parameters as a dictionary.
        api_key: Optional API key. Falls back to NEWS_API_KEY env var.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        NewsAPIError: If the request fails or the API returns an error.
    """
    key = api_key or NEWS_API_KEY
    if not key:
        raise NewsAPIError(
            "No API key provided. Set the NEWS_API_KEY environment variable "
            "or pass api_key to the function."
        )

    url = f"{NEWS_API_BASE_URL}/{endpoint}"
    params_encoded = urllib.parse.urlencode(params)
    full_url = f"{url}?{params_encoded}"

    req = urllib.request.Request(full_url)
    req.add_header("X-Api-Key", key)
    req.add_header("User-Agent", "fetch-news/1.0")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
            error_data = json.loads(error_body)
            error_msg = error_data.get("message", str(e))
        except Exception:
            error_msg = error_body or str(e)
        raise NewsAPIError(f"HTTP {e.code}: {error_msg}", status_code=e.code) from e
    except urllib.error.URLError as e:
        raise NewsAPIError(f"Connection error: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise NewsAPIError(f"Failed to parse API response: {e}") from e

    if data.get("status") != "ok":
        error_msg = data.get("message", "Unknown API error")
        raise NewsAPIError(f"API error: {error_msg}")

    return data


def fetch_news(
    query: Optional[str] = None,
    category: Optional[str] = None,
    country: str = "us",
    page_size: int = 10,
    page: int = 1,
    api_key: Optional[str] = None,
) -> NewsResponse:
    """
    Fetch news articles from the News API.

    This function retrieves news articles either by searching with a query
    string or by fetching top headlines filtered by category and/or country.

    Args:
        query: Search query string. If provided, searches all articles.
               If None, fetches top headlines.
        category: News category (e.g., 'business', 'technology', 'sports',
                  'science', 'health', 'entertainment', 'general').
                  Only used when fetching top headlines (query is None).
        country: Two-letter ISO 3166-1 country code (default: 'us').
                 Only used when fetching top headlines (query is None).
        page_size: Number of articles to return per page (default: 10, max: 100).
        page: Page number for pagination (default: 1).
        api_key: Optional API key. Falls back to NEWS_API_KEY env var.

    Returns:
        NewsResponse object containing status, total results, and a list
        of NewsArticle objects.

    Raises:
        NewsAPIError: If the API request fails or returns an error.
        ValueError: If invalid parameters are provided.

    Examples:
        # Fetch top headlines
        >>> response = fetch_news()
        >>> for article in response.articles:
        ...     print(article.title)

        # Search for specific news
        >>> response = fetch_news(query="artificial intelligence")

        # Fetch technology news
        >>> response = fetch_news(category="technology")
    """
    if page_size < 1 or page_size > 100:
        raise ValueError("page_size must be between 1 and 100")
    if page < 1:
        raise ValueError("page must be >= 1")

    valid_categories = {
        "business", "entertainment", "general", "health",
        "science", "sports", "technology"
    }
    if category and category.lower() not in valid_categories:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of: {', '.join(sorted(valid_categories))}"
        )

    params = {
        "pageSize": page_size,
        "page": page,
    }

    if query:
        # Use the 'everything' endpoint for search queries
        endpoint = "everything"
        params["q"] = query
    else:
        # Use the 'top-headlines' endpoint for browsing
        endpoint = "top-headlines"
        params["country"] = country
        if category:
            params["category"] = category.lower()

    data = _make_request(endpoint, params, api_key=api_key)

    articles = [_parse_article(a) for a in data.get("articles", [])]

    return NewsResponse(
        status=data.get("status", "ok"),
        total_results=data.get("totalResults", 0),
        articles=articles,
    )


def fetch_top_headlines(
    country: str = "us",
    category: Optional[str] = None,
    page_size: int = 10,
    api_key: Optional[str] = None,
) -> NewsResponse:
    """
    Convenience function to fetch top headlines.

    Args:
        country: Two-letter ISO 3166-1 country code (default: 'us').
        category: News category filter.
        page_size: Number of articles to return (default: 10).
        api_key: Optional API key.

    Returns:
        NewsResponse with top headline articles.
    """
    return fetch_news(
        country=country,
        category=category,
        page_size=page_size,
        api_key=api_key,
    )


def search_news(
    query: str,
    page_size: int = 10,
    page: int = 1,
    api_key: Optional[str] = None,
) -> NewsResponse:
    """
    Convenience function to search for news articles.

    Args:
        query: Search query string.
        page_size: Number of articles to return per page (default: 10).
        page: Page number for pagination (default: 1).
        api_key: Optional API key.

    Returns:
        NewsResponse with matching articles.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    return fetch_news(
        query=query,
        page_size=page_size,
        page=page,
        api_key=api_key,
    )


if __name__ == "__main__":
    # Simple CLI usage example
    import sys

    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:])
        print(f"Searching for: {search_query}")
        try:
            result = search_news(search_query)
            print(f"Found {result.total_results} results")
            for article in result.articles:
                print(f"  - {article.title} ({article.source})")
        except NewsAPIError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Fetching top headlines...")
        try:
            result = fetch_top_headlines()
            print(f"Found {result.total_results} results")
            for article in result.articles:
                print(f"  - {article.title} ({article.source})")
        except NewsAPIError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
