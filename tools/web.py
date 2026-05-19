"""
Web tools: search and fetch pages.

The goal is not to be a full browser. The tools should be predictable,
safe enough for agent use, and return text that is easy for the model to use.
"""

from __future__ import annotations

import html
import ipaddress
import socket
import urllib.parse
from html.parser import HTMLParser

import requests


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
DEFAULT_TIMEOUT = (5, 15)
DEFAULT_MAX_RESULTS = 5
DEFAULT_MAX_FETCH_CHARS = 4000
DEFAULT_MAX_FETCH_BYTES = 1_000_000
BLOCKED_HOSTS = {"localhost"}
BLOCKED_SUFFIXES = (".local",)
BLOCKED_SCHEMES = {"http", "https"}
BLOCKED_NETS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


def _request_headers() -> dict:
    return {"User-Agent": USER_AGENT}


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _safe_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme.lower() not in BLOCKED_SCHEMES:
        raise ValueError("only http and https URLs are allowed")
    if not parsed.netloc:
        raise ValueError("URL is missing a hostname")

    hostname = (parsed.hostname or "").lower()
    if hostname in BLOCKED_HOSTS or any(hostname.endswith(suf) for suf in BLOCKED_SUFFIXES):
        raise ValueError(f"blocked hostname: {hostname}")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None

    if ip is not None:
        if any(ip in net for net in BLOCKED_NETS) or ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise ValueError(f"blocked IP address: {hostname}")
        return urllib.parse.urlunsplit(parsed)

    try:
        infos = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"unable to resolve hostname: {hostname}") from exc

    resolved_ips = set()
    for info in infos:
        sockaddr = info[4][0]
        try:
            resolved_ips.add(ipaddress.ip_address(sockaddr))
        except ValueError:
            continue

    if not resolved_ips:
        raise ValueError(f"unable to resolve hostname: {hostname}")

    for resolved in resolved_ips:
        if any(resolved in net for net in BLOCKED_NETS) or resolved.is_loopback or resolved.is_private or resolved.is_link_local or resolved.is_multicast or resolved.is_reserved:
            raise ValueError(f"blocked resolved address: {resolved}")

    return urllib.parse.urlunsplit(parsed)


def _read_limited_response(resp: requests.Response, max_bytes: int = DEFAULT_MAX_FETCH_BYTES) -> str:
    chunks = []
    total = 0
    for chunk in resp.iter_content(chunk_size=8192):
        if not chunk:
            continue
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_bytes:
            break
    data = b"".join(chunks)
    encoding = resp.encoding or resp.apparent_encoding or "utf-8"
    return data.decode(encoding, errors="replace")


class _DuckDuckGoParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._current = None
        self._capture_title = False
        self._capture_snippet = False
        self._text_buffer = []

    def _flush_text(self) -> str:
        text = _clean_text(html.unescape(" ".join(self._text_buffer)))
        self._text_buffer = []
        return text

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        class_name = attrs.get("class", "")
        if tag == "a" and "result__a" in class_name:
            self._current = {
                "title": "",
                "url": attrs.get("href", ""),
                "snippet": "",
            }
            self._capture_title = True
        elif self._current is not None and "result__snippet" in class_name:
            self._capture_snippet = True

    def handle_endtag(self, tag):
        if tag == "a" and self._capture_title and self._current is not None:
            self._current["title"] = self._flush_text()
            self._capture_title = False
        elif self._capture_snippet and tag in {"a", "span", "div"} and self._current is not None:
            snippet = self._flush_text()
            if snippet and not self._current["snippet"]:
                self._current["snippet"] = snippet
            self._capture_snippet = False
        elif tag == "article" and self._current is not None:
            if self._current["title"] or self._current["snippet"]:
                self.results.append(self._current)
            self._current = None
            self._capture_title = False
            self._capture_snippet = False

    def handle_data(self, data):
        if self._capture_title or self._capture_snippet:
            stripped = data.strip()
            if stripped:
                self._text_buffer.append(stripped)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip_depth = 0
        self.skip_tags = {"script", "style", "noscript", "nav", "footer", "header", "aside", "svg"}
        self.block_tags = {
            "p", "div", "section", "article", "main", "br", "li",
            "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th",
        }
        self.title = []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.skip_depth += 1
        elif self.skip_depth == 0 and tag in self.block_tags:
            self.text.append("\n")
        elif self.skip_depth == 0 and tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self.skip_tags and self.skip_depth > 0:
            self.skip_depth -= 1
        elif self.skip_depth == 0 and tag in self.block_tags:
            self.text.append("\n")
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self.skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self.text.append(stripped)
                if self._in_title:
                    self.title.append(stripped)


def _normalize_duckduckgo_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    parsed = urllib.parse.urlsplit(url)
    if "duckduckgo.com/l/" in (parsed.netloc + parsed.path):
        query = urllib.parse.parse_qs(parsed.query)
        for key in ("uddg", "u"):
            if key in query and query[key]:
                return urllib.parse.unquote(query[key][0])
    return url


def web_search(query: str, count: int = DEFAULT_MAX_RESULTS) -> str:
    """Search the web for a query and return structured results."""
    try:
        count = max(1, min(int(count), 10))
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "num": count},
            headers=_request_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()

        parser = _DuckDuckGoParser()
        parser.feed(resp.text)

        lines = [f"Search results for: {query}"]
        results = parser.results[:count]
        if not results:
            lines.append("No structured results were parsed from the search page.")
            return "\n".join(lines)

        for idx, result in enumerate(results, 1):
            title = _clean_text(result.get("title", "")) or "(untitled)"
            url = _normalize_duckduckgo_url(result.get("url", ""))
            snippet = _clean_text(result.get("snippet", "")) or "(no snippet)"
            lines.append(f"{idx}. {title}")
            lines.append(f"   url: {url}")
            lines.append(f"   snippet: {snippet}")

        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def web_fetch(url: str, max_chars: int = DEFAULT_MAX_FETCH_CHARS) -> str:
    """Fetch and extract text content from a URL with SSRF safeguards."""
    try:
        safe_url = _safe_url(url)
        resp = requests.get(
            safe_url,
            headers=_request_headers(),
            timeout=DEFAULT_TIMEOUT,
            allow_redirects=True,
            stream=True,
        )
        resp.raise_for_status()

        final_url = _safe_url(resp.url)
        content_type = resp.headers.get("content-type", "").lower()
        if content_type and not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml+xml")):
            return f"Fetch error: unsupported content type: {content_type}"

        text = _read_limited_response(resp)
        parser = _TextExtractor()
        parser.feed(text)
        content = _clean_text(" ".join(parser.text))

        if len(content) > max_chars:
            content = content[:max_chars] + "\n[...truncated]"

        title = _clean_text(" ".join(parser.title))
        lines = [f"Content from {final_url}"]
        if title:
            lines.append(f"title: {title}")
        lines.append(content or "(no readable text found)")
        return "\n".join(lines)
    except Exception as e:
        return f"Fetch error: {e}"
