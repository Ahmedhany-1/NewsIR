"""
Search Engine Module for News IR System
=========================================
Handles query processing, search execution, and result formatting.
Supports keyword highlighting in titles and snippets.
"""

import re
from preprocessor import TextPreprocessor
from indexer import InvertedIndex


class SearchEngine:
    """
    Search engine that processes queries and returns ranked results.
    Supports TF-IDF, BM25, and Boolean search modes.
    """

    def __init__(self, index: InvertedIndex, preprocessor: TextPreprocessor = None):
        self.index = index
        self.preprocessor = preprocessor or TextPreprocessor()

    def search(self, query, mode="bm25", top_k=10):
        """
        Execute a search query.

        Args:
            query: Raw query string
            mode: 'bm25', 'tfidf', or 'boolean'
            top_k: Number of results to return

        Returns:
            List of result dicts with doc info and scores
        """
        # Preprocess query and keep original words for snippet matching
        query_tokens, query_original_words = self.preprocessor.get_original_words_for_query(query)

        if not query_tokens:
            return []

        # Execute search based on mode
        if mode == "bm25":
            ranked = self.index.search_bm25(query_tokens, top_k)
        elif mode == "tfidf":
            ranked = self.index.search_tfidf(query_tokens, top_k)
        elif mode == "boolean":
            doc_ids = self.index.boolean_search(query_tokens, "AND")
            ranked = []
            for doc_id in doc_ids:
                score = self.index.bm25_score(query_tokens, doc_id)
                ranked.append((doc_id, score))
            ranked.sort(key=lambda x: x[1], reverse=True)
            ranked = ranked[:top_k]
        else:
            ranked = self.index.search_bm25(query_tokens, top_k)

        # Build the set of highlight words (original + stemmed for broader matching)
        highlight_words = set(w.lower() for w in query_original_words)
        highlight_stems = set(query_tokens)

        # Format results
        results = []
        for rank, (doc_id, score) in enumerate(ranked, 1):
            doc = self.index.get_document(doc_id)
            if doc:
                snippet = self._generate_snippet(doc_id, query_original_words, query_tokens)
                highlighted_title = self._highlight_text(doc["title"], highlight_words, highlight_stems)
                highlighted_snippet = self._highlight_text(snippet, highlight_words, highlight_stems)

                results.append({
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": round(score, 4),
                    "title": doc["title"],
                    "title_highlighted": highlighted_title,
                    "snippet": snippet,
                    "snippet_highlighted": highlighted_snippet,
                    "source": doc["source"],
                    "url": doc["url"],
                    "published_date": doc["published_date"],
                })

        return results

    def _highlight_text(self, text, highlight_words, highlight_stems):
        """
        Highlight matching query words in text by wrapping them in <mark> tags.
        
        Matches on:
          1. Exact original query words (e.g. "intelligence" highlights "intelligence")
          2. Words whose stem matches a query stem (e.g. query "technology" stems to
             "technolog", which also matches "technologies", "technological", etc.)
        """
        if not text or (not highlight_words and not highlight_stems):
            return text

        stemmer = self.preprocessor.stemmer

        # Split text into tokens while preserving separators (spaces, punctuation)
        # This regex splits on word boundaries, keeping all parts
        parts = re.split(r'(\b\w+\b)', text)

        result = []
        for part in parts:
            if not part:
                continue
            # Check if this part is a word
            if re.match(r'^\w+$', part):
                word_lower = part.lower()
                matched = False

                # Check 1: exact match with original query words
                if word_lower in highlight_words:
                    matched = True

                # Check 2: stem match — stem the word and see if it matches any query stem
                if not matched:
                    try:
                        word_stem = stemmer.stem(word_lower)
                        if word_stem in highlight_stems:
                            matched = True
                    except Exception:
                        pass

                if matched:
                    result.append(f'<mark>{part}</mark>')
                else:
                    result.append(part)
            else:
                result.append(part)

        return ''.join(result)

    def _generate_snippet(self, doc_id, query_original_words, query_stemmed_tokens, max_length=300):
        """
        Generate a relevant text snippet for a document.
        
        Uses ORIGINAL query words (not stemmed) to find matching sentences
        in the original document content, so snippets are actually relevant.
        """
        doc = self.index.get_document(doc_id)
        if not doc:
            return ""

        content = doc.get("original_content", "")
        if not content:
            return ""

        # Split into sentences properly
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(content)
        except Exception:
            sentences = re.split(r'[.!?]+\s+', content)

        if not sentences:
            return content[:max_length] + "..."

        # Score each sentence by how many ORIGINAL query words it contains
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0
            for word in query_original_words:
                if word.lower() in sentence_lower:
                    score += 2
            for stem in query_stemmed_tokens:
                if len(stem) >= 3 and stem in sentence_lower:
                    score += 1
            scored_sentences.append((score, sentence))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if scored_sentences[0][0] > 0:
            snippet_parts = []
            total_len = 0
            for score, sent in scored_sentences:
                if score == 0:
                    break
                if total_len + len(sent) > max_length:
                    if not snippet_parts:
                        snippet_parts.append(sent[:max_length])
                    break
                snippet_parts.append(sent)
                total_len += len(sent)
            snippet = " ".join(snippet_parts)
        else:
            snippet = content[:max_length]

        if len(snippet) < len(content):
            snippet += "..."

        return snippet

    def display_results(self, results, query):
        """Display search results in a formatted way."""
        print(f"\n{'=' * 60}")
        print(f"  Search Results for: \"{query}\"")
        print(f"  Found {len(results)} results")
        print(f"{'=' * 60}")

        if not results:
            print("  No results found.")
            return

        for r in results:
            print(f"\n  [{r['rank']}] {r['title']}")
            print(f"      Score:  {r['score']}")
            print(f"      Source: {r['source']}")
            print(f"      URL:    {r['url']}")
            print(f"      {r['snippet']}")
            print(f"      {'-' * 50}")
