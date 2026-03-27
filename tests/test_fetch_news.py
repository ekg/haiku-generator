"""Tests for the fetch_news module."""

import json
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fetch_news import (
    fetch_news,
    fetch_top_headlines,
    search_news,
    NewsArticle,
    NewsResponse,
    NewsAPIError,
    _parse_article,
)


class TestNewsArticle(unittest.TestCase):
    """Tests for NewsArticle class."""

    def test_article_creation(self):
        article = NewsArticle(
            title="Test Title",
            description="Test Description",
            url="https://example.com/article",
            source="Test Source",
            published_at="2024-01-01T00:00:00Z",
            author="Test Author",
        )
        self.assertEqual(article.title, "Test Title")
        self.assertEqual(article.source, "Test Source")
        self.assertEqual(article.author, "Test Author")

    def test_article_to_dict(self):
        article = NewsArticle(
            title="Test",
            description="Desc",
            url="https://example.com",
            source="Src",
            published_at="2024-01-01",
        )
        d = article.to_dict()
        self.assertEqual(d["title"], "Test")
        self.assertEqual(d["source"], "Src")
        self.assertIsNone(d["author"])
        self.assertIsNone(d["image_url"])

    def test_article_repr(self):
        article = NewsArticle(
            title="A" * 100,
            description="",
            url="",
            source="TestSrc",
            published_at="",
        )
        r = repr(article)
        self.assertIn("TestSrc", r)
        self.assertIn("...", r)


class TestParseArticle(unittest.TestCase):
    """Tests for _parse_article helper."""

    def test_parse_full_article(self):
        data = {
            "source": {"id": "test", "name": "Test News"},
            "title": "Article Title",
            "description": "Article Description",
            "url": "https://example.com/article",
            "publishedAt": "2024-01-01T12:00:00Z",
            "author": "John Doe",
            "urlToImage": "https://example.com/image.jpg",
            "content": "Full article content...",
        }
        article = _parse_article(data)
        self.assertEqual(article.title, "Article Title")
        self.assertEqual(article.source, "Test News")
        self.assertEqual(article.author, "John Doe")
        self.assertEqual(article.image_url, "https://example.com/image.jpg")

    def test_parse_minimal_article(self):
        data = {}
        article = _parse_article(data)
        self.assertEqual(article.title, "")
        self.assertEqual(article.source, "Unknown")
        self.assertIsNone(article.author)

    def test_parse_article_with_string_source(self):
        data = {"source": "Simple Source"}
        article = _parse_article(data)
        self.assertEqual(article.source, "Simple Source")


class TestFetchNewsValidation(unittest.TestCase):
    """Tests for parameter validation in fetch_news."""

    def test_invalid_page_size_too_small(self):
        with self.assertRaises(ValueError):
            fetch_news(page_size=0, api_key="test-key")

    def test_invalid_page_size_too_large(self):
        with self.assertRaises(ValueError):
            fetch_news(page_size=101, api_key="test-key")

    def test_invalid_page(self):
        with self.assertRaises(ValueError):
            fetch_news(page=0, api_key="test-key")

    def test_invalid_category(self):
        with self.assertRaises(ValueError):
            fetch_news(category="invalid_category", api_key="test-key")

    def test_valid_categories(self):
        valid = ["business", "entertainment", "general", "health",
                 "science", "sports", "technology"]
        for cat in valid:
            # Should not raise - will fail at API level since we're not mocking,
            # but validation should pass
            try:
                fetch_news(category=cat, api_key="test-key")
            except NewsAPIError:
                pass  # Expected - no real API call
            except ValueError:
                self.fail(f"Category '{cat}' should be valid")


class TestSearchNewsValidation(unittest.TestCase):
    """Tests for parameter validation in search_news."""

    def test_empty_query(self):
        with self.assertRaises(ValueError):
            search_news("", api_key="test-key")

    def test_whitespace_query(self):
        with self.assertRaises(ValueError):
            search_news("   ", api_key="test-key")


class TestNoApiKey(unittest.TestCase):
    """Tests for missing API key handling."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_api_key_raises_error(self):
        # Remove NEWS_API_KEY from environment
        with patch('fetch_news.NEWS_API_KEY', ''):
            with self.assertRaises(NewsAPIError) as ctx:
                fetch_news()
            self.assertIn("No API key", str(ctx.exception))


class TestNewsResponse(unittest.TestCase):
    """Tests for NewsResponse class."""

    def test_response_to_dict(self):
        articles = [
            NewsArticle("T1", "D1", "http://u1", "S1", "2024-01-01"),
            NewsArticle("T2", "D2", "http://u2", "S2", "2024-01-02"),
        ]
        response = NewsResponse(status="ok", total_results=2, articles=articles)
        d = response.to_dict()
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["total_results"], 2)
        self.assertEqual(len(d["articles"]), 2)
        self.assertEqual(d["articles"][0]["title"], "T1")


class TestFetchNewsWithMock(unittest.TestCase):
    """Tests for fetch_news with mocked HTTP requests."""

    def _mock_api_response(self, data):
        """Create a mock response object."""
        response = MagicMock()
        response.read.return_value = json.dumps(data).encode("utf-8")
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        return response

    @patch("fetch_news.urllib.request.urlopen")
    def test_fetch_top_headlines_success(self, mock_urlopen):
        api_response = {
            "status": "ok",
            "totalResults": 1,
            "articles": [
                {
                    "source": {"id": "bbc", "name": "BBC News"},
                    "title": "Breaking News",
                    "description": "Something happened",
                    "url": "https://bbc.co.uk/news/1",
                    "publishedAt": "2024-01-01T00:00:00Z",
                    "author": "BBC",
                    "urlToImage": None,
                    "content": "Full story...",
                }
            ],
        }
        mock_urlopen.return_value = self._mock_api_response(api_response)

        result = fetch_news(api_key="test-key-123")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.total_results, 1)
        self.assertEqual(len(result.articles), 1)
        self.assertEqual(result.articles[0].title, "Breaking News")
        self.assertEqual(result.articles[0].source, "BBC News")

    @patch("fetch_news.urllib.request.urlopen")
    def test_search_news_success(self, mock_urlopen):
        api_response = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "source": {"name": "Tech News"},
                    "title": "AI Breakthrough",
                    "description": "New AI development",
                    "url": "https://technews.com/1",
                    "publishedAt": "2024-06-15T10:00:00Z",
                },
                {
                    "source": {"name": "Science Daily"},
                    "title": "AI in Medicine",
                    "description": "AI helps doctors",
                    "url": "https://sciencedaily.com/1",
                    "publishedAt": "2024-06-14T08:00:00Z",
                },
            ],
        }
        mock_urlopen.return_value = self._mock_api_response(api_response)

        result = search_news("artificial intelligence", api_key="test-key-123")

        self.assertEqual(result.total_results, 2)
        self.assertEqual(len(result.articles), 2)
        self.assertEqual(result.articles[0].title, "AI Breakthrough")

    @patch("fetch_news.urllib.request.urlopen")
    def test_fetch_with_category(self, mock_urlopen):
        api_response = {
            "status": "ok",
            "totalResults": 0,
            "articles": [],
        }
        mock_urlopen.return_value = self._mock_api_response(api_response)

        result = fetch_top_headlines(category="technology", api_key="test-key")

        self.assertEqual(result.total_results, 0)
        self.assertEqual(len(result.articles), 0)
        # Verify the URL used contains category and top-headlines endpoint
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertIn("top-headlines", request_obj.full_url)
        self.assertIn("category=technology", request_obj.full_url)


if __name__ == "__main__":
    unittest.main()
