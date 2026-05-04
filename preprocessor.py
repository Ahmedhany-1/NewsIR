"""
Text Preprocessing Module for News IR System
==============================================
Handles: Tokenization, Stemming, Lemmatization, Stop Word Removal, Text Normalization
"""

import re
import json
import os
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextPreprocessor:
    """
    Comprehensive text preprocessor for IR system.
    Supports tokenization, normalization, stop word removal,
    stemming, and lemmatization.
    """

    def __init__(self, use_stemming=True, use_lemmatization=True, remove_stopwords=True):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        # Add custom stop words common in news articles
        self.stop_words.update([
            "said", "also", "would", "could", "one", "two", "new",
            "like", "may", "us", "get", "make", "know", "say",
            "reuters", "bbc", "cnn", "guardian", "npr", "ap",
        ])
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords

        # Preprocessing statistics
        self.stats = {
            "total_docs_processed": 0,
            "total_tokens_before": 0,
            "total_tokens_after": 0,
            "avg_doc_length_before": 0,
            "avg_doc_length_after": 0,
        }

    def normalize_text(self, text):
        """
        Text normalization:
        - Convert to lowercase
        - Remove URLs, emails, HTML tags
        - Remove special characters and numbers
        - Normalize whitespace
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove special characters but keep apostrophes in contractions
        text = re.sub(r"[^a-zA-Z\s']", " ", text)

        # Remove standalone apostrophes
        text = re.sub(r"\s'|'\s", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text):
        """Tokenize text into words using NLTK word tokenizer."""
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]

    def remove_stop_words(self, tokens):
        """Remove stop words from token list."""
        return [t for t in tokens if t not in self.stop_words]

    def stem_tokens(self, tokens):
        """Apply Porter stemming to tokens."""
        return [self.stemmer.stem(t) for t in tokens]

    def lemmatize_tokens(self, tokens):
        """Apply WordNet lemmatization to tokens."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text):
        """
        Full preprocessing pipeline:
        1. Text Normalization
        2. Tokenization
        3. Stop Word Removal
        4. Lemmatization (reduces to dictionary form)
        5. Stemming (reduces to root form)

        NOTE: We apply lemmatization first, then stemming.
        Stemming is the final step so the index uses stemmed forms.

        Returns processed token list.
        """
        # Step 1: Normalize
        normalized = self.normalize_text(text)

        # Step 2: Tokenize
        tokens = self.tokenize(normalized)
        tokens_before = len(tokens)

        # Step 3: Remove stop words
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)

        # Step 4: Lemmatize
        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)

        # Step 5: Stem
        if self.use_stemming:
            tokens = self.stem_tokens(tokens)

        # Update stats
        self.stats["total_tokens_before"] += tokens_before
        self.stats["total_tokens_after"] += len(tokens)

        return tokens

    def preprocess_keep_original(self, text):
        """
        Preprocess text but also return a mapping from stems back to
        original normalized words. Used for snippet highlighting.
        
        Returns:
            (tokens, stem_to_original_map)
        """
        normalized = self.normalize_text(text)
        tokens = self.tokenize(normalized)
        
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        original_tokens = list(tokens)  # Save before stemming/lemmatization
        
        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        if self.use_stemming:
            tokens = self.stem_tokens(tokens)
        
        # Build reverse map: stemmed -> set of originals
        stem_to_original = {}
        for orig, stemmed in zip(original_tokens, tokens):
            if stemmed not in stem_to_original:
                stem_to_original[stemmed] = set()
            stem_to_original[stemmed].add(orig)
        
        return tokens, stem_to_original

    def get_original_words_for_query(self, query):
        """
        For a query string, return both the processed tokens AND the
        original normalized words (for matching in snippets).
        """
        normalized = self.normalize_text(query)
        original_words = self.tokenize(normalized)
        if self.remove_stopwords:
            original_words = self.remove_stop_words(original_words)
        
        processed = list(original_words)
        if self.use_lemmatization:
            processed = self.lemmatize_tokens(processed)
        if self.use_stemming:
            processed = self.stem_tokens(processed)
        
        return processed, original_words

    def preprocess_documents(self, articles):
        """
        Preprocess a collection of news articles.

        Args:
            articles: List of article dicts with 'title', 'content', 'doc_id'

        Returns:
            List of processed document dicts
        """
        processed_docs = []
        print("\n" + "=" * 60)
        print("  TEXT PREPROCESSING")
        print("=" * 60)

        for i, article in enumerate(articles):
            doc_id = article["doc_id"]
            title = article.get("title", "")
            content = article.get("content", "")

            # Preprocess title and content separately
            title_tokens = self.preprocess(title)
            content_tokens = self.preprocess(content)

            # Title boosting: add title tokens with higher weight (3x)
            tokens = title_tokens * 3 + content_tokens

            if len(tokens) < 5:
                continue

            processed_doc = {
                "doc_id": doc_id,
                "title": title,
                "original_content": content[:2000],  # Store more content for better snippets
                "tokens": tokens,
                "token_count": len(tokens),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "published_date": article.get("published_date", ""),
            }
            processed_docs.append(processed_doc)

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Processed {i + 1}/{len(articles)} documents...")

        self.stats["total_docs_processed"] = len(processed_docs)
        if processed_docs:
            self.stats["avg_doc_length_before"] = round(
                self.stats["total_tokens_before"] / len(processed_docs), 1
            )
            self.stats["avg_doc_length_after"] = round(
                self.stats["total_tokens_after"] / len(processed_docs), 1
            )

        print(f"\n  Preprocessing Statistics:")
        print(f"  - Documents processed: {self.stats['total_docs_processed']}")
        print(f"  - Avg tokens before:   {self.stats['avg_doc_length_before']}")
        print(f"  - Avg tokens after:    {self.stats['avg_doc_length_after']}")
        reduction = 0
        if self.stats['total_tokens_before'] > 0:
            reduction = (1 - self.stats['total_tokens_after'] / self.stats['total_tokens_before']) * 100
        print(f"  - Token reduction:     {reduction:.1f}%")
        print("=" * 60)

        return processed_docs

    def get_vocabulary_stats(self, processed_docs):
        """Get vocabulary statistics from processed documents."""
        all_tokens = []
        for doc in processed_docs:
            all_tokens.extend(doc["tokens"])

        vocab = set(all_tokens)
        freq = Counter(all_tokens)
        most_common = freq.most_common(30)

        return {
            "vocabulary_size": len(vocab),
            "total_tokens": len(all_tokens),
            "most_common_30": most_common,
            "avg_frequency": round(len(all_tokens) / max(len(vocab), 1), 2),
        }


def save_processed(processed_docs, filepath="data/processed_articles.json"):
    """Save processed documents."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved {len(processed_docs)} processed documents to {filepath}")


def load_processed(filepath="data/processed_articles.json"):
    """Load processed documents."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
