"""
Data Collection Module for News IR System
==========================================
Collects news articles from real-world sources using RSS feeds and web scraping.
Sources: BBC News, Reuters, CNN, Al Jazeera, The Guardian, NPR
"""

import os
import json
import time
import hashlib
import requests
import feedparser
from datetime import datetime
from bs4 import BeautifulSoup

# RSS Feed sources for news articles
RSS_FEEDS = {
    "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
    "BBC Technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Science": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "Reuters Top News": "https://feeds.reuters.com/reuters/topNews",
    "Reuters Technology": "https://feeds.reuters.com/reuters/technologyNews",
    "NPR News": "https://feeds.npr.org/1001/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "The Guardian World": "https://www.theguardian.com/world/rss",
    "The Guardian Tech": "https://www.theguardian.com/technology/rss",
    "CNN Top Stories": "http://rss.cnn.com/rss/edition.rss",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def generate_doc_id(url):
    """Generate a unique document ID from URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def fetch_article_content(url, timeout=10):
    """Fetch and extract the main text content from a news article URL."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        # Try to find article body using common selectors
        article_body = None
        selectors = [
            "article",
            '[role="main"]',
            ".article-body",
            ".story-body",
            ".article__body",
            ".post-content",
            ".entry-content",
            "#article-body",
            ".content__article-body",
        ]
        for selector in selectors:
            article_body = soup.select_one(selector)
            if article_body:
                break

        if not article_body:
            article_body = soup.find("body")

        if article_body:
            paragraphs = article_body.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)
            return text if len(text) > 100 else None
        return None

    except Exception as e:
        print(f"  [WARN] Could not fetch {url}: {e}")
        return None


def collect_from_rss(max_per_feed=15, fetch_full_text=True):
    """
    Collect news articles from RSS feeds.
    
    Args:
        max_per_feed: Maximum articles to collect per feed
        fetch_full_text: Whether to fetch full article text from URLs
        
    Returns:
        List of article dictionaries
    """
    articles = []
    seen_urls = set()

    print("=" * 60)
    print("  NEWS DATA COLLECTION")
    print("=" * 60)

    for source_name, feed_url in RSS_FEEDS.items():
        print(f"\n[*] Fetching from: {source_name}")
        try:
            feed = feedparser.parse(feed_url)
            entries = feed.entries[:max_per_feed]
            count = 0

            for entry in entries:
                url = entry.get("link", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                title = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()

                # Clean HTML from summary
                if summary:
                    summary = BeautifulSoup(summary, "html.parser").get_text(strip=True)

                published = entry.get("published", entry.get("updated", ""))

                # Fetch full article text
                full_text = None
                if fetch_full_text and url:
                    full_text = fetch_article_content(url)
                    time.sleep(0.5)  # Be polite to servers

                # Use full text if available, otherwise fall back to summary
                content = full_text if full_text else summary

                if not title or not content or len(content) < 50:
                    continue

                doc_id = generate_doc_id(url)
                article = {
                    "doc_id": doc_id,
                    "title": title,
                    "content": content,
                    "summary": summary[:500] if summary else "",
                    "source": source_name,
                    "url": url,
                    "published_date": published,
                    "collected_at": datetime.now().isoformat(),
                    "has_full_text": full_text is not None,
                }
                articles.append(article)
                count += 1
                print(f"  [{count}] {title[:60]}...")

            print(f"  -> Collected {count} articles from {source_name}")

        except Exception as e:
            print(f"  [ERROR] Failed to fetch {source_name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  TOTAL ARTICLES COLLECTED: {len(articles)}")
    print(f"{'=' * 60}")
    return articles


def save_articles(articles, filepath="data/raw_articles.json"):
    """Save collected articles to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved {len(articles)} articles to {filepath}")
    return filepath


def load_articles(filepath="data/raw_articles.json"):
    """Load articles from JSON file."""
    if not os.path.exists(filepath):
        print(f"[!] File not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"[+] Loaded {len(articles)} articles from {filepath}")
    return articles


if __name__ == "__main__":
    articles = collect_from_rss(max_per_feed=15, fetch_full_text=True)
    save_articles(articles)
