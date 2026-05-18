"""
Web tools: search and fetch pages.
"""

import urllib.parse
from typing import Optional

import requests

def web_search(query: str, count: int = 5) -> str:
    """Search the web for a query. Returns a list of results with titles and snippets."""
    try:
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}&num={min(count, 10)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        # Simple extraction — in production use BeautifulSoup
        from html.parser import HTMLParser

        class SimpleExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.capture = False
                self.skip_tags = {"script", "style", "noscript"}

            def handle_starttag(self, tag, attrs):
                if tag in self.skip_tags:
                    self.capture = False

            def handle_endtag(self, tag):
                if tag in self.skip_tags:
                    self.capture = True

            def handle_data(self, data):
                stripped = data.strip()
                if stripped:
                    self.text.append(stripped)

        extractor = SimpleExtractor()
        extractor.feed(resp.text)
        content = " ".join(extractor.text[:200])

        return f"Search results for '{query}':\n{content[:2000]}"
    except Exception as e:
        return f"Search error: {e}"


def web_fetch(url: str, max_chars: int = 4000) -> str:
    """Fetch and extract text content from a URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.skip = False
                self.skip_tags = {"script", "style", "noscript", "nav", "footer", "header"}

            def handle_starttag(self, tag, attrs):
                if tag in self.skip_tags:
                    self.skip = True

            def handle_endtag(self, tag):
                if tag in self.skip_tags:
                    self.skip = False

            def handle_data(self, data):
                if not self.skip:
                    stripped = data.strip()
                    if stripped:
                        self.text.append(stripped)

        extractor = TextExtractor()
        extractor.feed(resp.text)
        content = "\n".join(extractor.text)

        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[...truncated]"

        return f"Content from {url}:\n{content}"
    except Exception as e:
        return f"Fetch error: {e}"
