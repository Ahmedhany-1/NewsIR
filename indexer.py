"""
Indexing Module for News IR System
====================================
Builds and manages the Inverted Index for efficient document retrieval.
Supports TF-IDF weighting and BM25 scoring.
"""

import os
import json
import math
import pickle
from collections import defaultdict, Counter


class InvertedIndex:
    """
    Inverted Index with TF-IDF and BM25 scoring.
    
    Structure:
        index[term] = {doc_id: term_frequency, ...}
    """

    def __init__(self):
        # Core index: term -> {doc_id: tf, ...}
        self.index = defaultdict(dict)
        # Document metadata
        self.documents = {}  # doc_id -> {title, source, url, ...}
        self.doc_lengths = {}  # doc_id -> number of tokens
        self.doc_tokens = {}  # doc_id -> token list (for snippets)
        # Corpus statistics
        self.total_docs = 0
        self.avg_doc_length = 0
        self.vocabulary = set()
        # IDF cache
        self._idf_cache = {}

    def build_index(self, processed_docs):
        """
        Build the inverted index from processed documents.
        
        Args:
            processed_docs: List of dicts with 'doc_id', 'tokens', 'title', etc.
        """
        print("\n" + "=" * 60)
        print("  BUILDING INVERTED INDEX")
        print("=" * 60)

        self.index.clear()
        self.documents.clear()
        self.doc_lengths.clear()

        total_length = 0

        for doc in processed_docs:
            doc_id = doc["doc_id"]
            tokens = doc["tokens"]

            # Store document metadata
            self.documents[doc_id] = {
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "url": doc.get("url", ""),
                "original_content": doc.get("original_content", ""),
                "published_date": doc.get("published_date", ""),
                "token_count": len(tokens),
            }

            self.doc_lengths[doc_id] = len(tokens)
            self.doc_tokens[doc_id] = tokens
            total_length += len(tokens)

            # Count term frequencies
            term_freq = Counter(tokens)

            # Add to inverted index
            for term, freq in term_freq.items():
                self.index[term][doc_id] = freq
                self.vocabulary.add(term)

        self.total_docs = len(self.documents)
        self.avg_doc_length = total_length / max(self.total_docs, 1)

        # Precompute IDF values
        self._compute_idf()

        print(f"  - Documents indexed:   {self.total_docs}")
        print(f"  - Vocabulary size:     {len(self.vocabulary)}")
        print(f"  - Avg document length: {self.avg_doc_length:.1f} tokens")
        print(f"  - Total postings:      {sum(len(v) for v in self.index.values())}")
        print("=" * 60)

    def _compute_idf(self):
        """Precompute IDF for all terms."""
        self._idf_cache = {}
        for term in self.index:
            df = len(self.index[term])
            self._idf_cache[term] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

    def get_idf(self, term):
        """Get IDF value for a term."""
        return self._idf_cache.get(term, 0)

    def get_tf(self, term, doc_id):
        """Get term frequency in a document."""
        return self.index.get(term, {}).get(doc_id, 0)

    def get_tfidf(self, term, doc_id):
        """Compute TF-IDF score for a term in a document."""
        tf = self.get_tf(term, doc_id)
        if tf == 0:
            return 0
        # Log-normalized TF
        tf_norm = 1 + math.log(tf) if tf > 0 else 0
        idf = self.get_idf(term)
        return tf_norm * idf

    def bm25_score(self, query_terms, doc_id, k1=1.5, b=0.75):
        """
        Compute BM25 score for a document given query terms.
        
        Args:
            query_terms: List of preprocessed query terms
            doc_id: Document ID
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            
        Returns:
            BM25 score
        """
        score = 0
        doc_len = self.doc_lengths.get(doc_id, 0)

        for term in query_terms:
            if term not in self.index or doc_id not in self.index[term]:
                continue

            tf = self.index[term][doc_id]
            idf = self.get_idf(term)

            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / self.avg_doc_length)
            score += idf * (numerator / denominator)

        return score

    def search_tfidf(self, query_terms, top_k=10):
        """
        Search using TF-IDF cosine similarity.
        
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        scores = defaultdict(float)
        
        # Get candidate documents (any doc containing at least one query term)
        candidate_docs = set()
        for term in query_terms:
            if term in self.index:
                candidate_docs.update(self.index[term].keys())

        for doc_id in candidate_docs:
            for term in query_terms:
                scores[doc_id] += self.get_tfidf(term, doc_id)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search_bm25(self, query_terms, top_k=10):
        """
        Search using BM25 ranking.
        
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        scores = {}

        # Get candidate documents
        candidate_docs = set()
        for term in query_terms:
            if term in self.index:
                candidate_docs.update(self.index[term].keys())

        for doc_id in candidate_docs:
            scores[doc_id] = self.bm25_score(query_terms, doc_id)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def boolean_search(self, query_terms, mode="AND"):
        """
        Boolean search (AND/OR).
        
        Returns:
            Set of matching document IDs
        """
        if not query_terms:
            return set()

        if mode == "AND":
            result = None
            for term in query_terms:
                docs = set(self.index.get(term, {}).keys())
                result = docs if result is None else result & docs
            return result or set()
        else:  # OR
            result = set()
            for term in query_terms:
                result.update(self.index.get(term, {}).keys())
            return result

    def get_document(self, doc_id):
        """Get document metadata by ID."""
        return self.documents.get(doc_id)

    def get_posting_list(self, term):
        """Get posting list for a term."""
        return self.index.get(term, {})

    def save_index(self, filepath="data/inverted_index.pkl"):
        """Save the inverted index to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "index": dict(self.index),
            "documents": self.documents,
            "doc_lengths": self.doc_lengths,
            "doc_tokens": self.doc_tokens,
            "total_docs": self.total_docs,
            "avg_doc_length": self.avg_doc_length,
            "vocabulary": list(self.vocabulary),
            "idf_cache": self._idf_cache,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[+] Index saved to {filepath}")

    def load_index(self, filepath="data/inverted_index.pkl"):
        """Load the inverted index from disk."""
        if not os.path.exists(filepath):
            print(f"[!] Index file not found: {filepath}")
            return False
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.index = defaultdict(dict, data["index"])
        self.documents = data["documents"]
        self.doc_lengths = data["doc_lengths"]
        self.doc_tokens = data.get("doc_tokens", {})
        self.total_docs = data["total_docs"]
        self.avg_doc_length = data["avg_doc_length"]
        self.vocabulary = set(data["vocabulary"])
        self._idf_cache = data["idf_cache"]
        print(f"[+] Index loaded: {self.total_docs} docs, {len(self.vocabulary)} terms")
        return True

    def get_index_stats(self):
        """Return index statistics."""
        return {
            "total_documents": self.total_docs,
            "vocabulary_size": len(self.vocabulary),
            "avg_document_length": round(self.avg_doc_length, 1),
            "total_postings": sum(len(v) for v in self.index.values()),
            "sources": list(set(d.get("source", "") for d in self.documents.values())),
        }
