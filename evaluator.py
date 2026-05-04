"""
Evaluation Module for News IR System
======================================
Evaluates IR system quality using Precision, Recall, F1 Score,
Mean Average Precision (MAP), and Normalized Discounted Cumulative Gain (NDCG).

Uses keyword-based relevance judgments: a document is relevant to a query
if its TITLE or CONTENT contains any of the specified relevance keywords.
"""

import json
import os
import re
import math
from collections import defaultdict


class IREvaluator:
    """
    Evaluates IR system performance using standard metrics.
    Uses keyword-based relevance judgments for test queries.
    """

    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.results = {}

    def create_test_queries(self, index):
        """
        Create test queries with ground-truth relevance judgments.
        
        Relevance is determined by checking if a document's TITLE or CONTENT
        contains specific keywords related to the query topic. This provides
        honest, content-based relevance judgments rather than circular 
        pseudo-relevance from the ranking algorithm itself.
        """
        # Each test query has:
        #   - query: the search query string
        #   - description: what the query is about
        #   - relevance_keywords: words that MUST appear in a relevant doc's
        #     title or content (at least one keyword must match)
        test_query_definitions = [
            {
                "query": "technology artificial intelligence",
                "description": "AI and tech news",
                "relevance_keywords": ["artificial intelligence", "ai ", " ai,", "machine learning",
                                       "neural network", "deep learning", "chatbot", "openai",
                                       "google ai", "robot"],
            },
            {
                "query": "climate change environment",
                "description": "Environmental and climate news",
                "relevance_keywords": ["climate", "global warming", "carbon", "emission",
                                       "greenhouse", "environment", "renewable energy",
                                       "fossil fuel", "temperature rise", "sea level"],
            },
            {
                "query": "economic growth market",
                "description": "Economy and financial markets",
                "relevance_keywords": ["economy", "economic", "gdp", "stock market", "inflation",
                                       "interest rate", "trade", "recession", "financial",
                                       "growth rate", "investor"],
            },
            {
                "query": "government policy election",
                "description": "Political news and elections",
                "relevance_keywords": ["election", "government", "president", "minister",
                                       "parliament", "vote", "political", "policy",
                                       "legislation", "democrat", "republican", "party"],
            },
            {
                "query": "health medical research",
                "description": "Health and medical news",
                "relevance_keywords": ["health", "medical", "doctor", "hospital", "patient",
                                       "disease", "treatment", "drug", "vaccine", "clinical",
                                       "symptom", "diagnosis", "cancer"],
            },
            {
                "query": "war conflict military",
                "description": "Military and conflict news",
                "relevance_keywords": ["war", "military", "army", "soldier", "conflict",
                                       "attack", "weapon", "troops", "invasion", "ceasefire",
                                       "defense", "nato", "combat"],
            },
            {
                "query": "sports football championship",
                "description": "Sports news",
                "relevance_keywords": ["sport", "football", "soccer", "championship", "league",
                                       "match", "team", "player", "score", "tournament",
                                       "olympic", "coach", "game"],
            },
            {
                "query": "education university student",
                "description": "Education news",
                "relevance_keywords": ["education", "university", "student", "school", "teacher",
                                       "college", "academic", "degree", "tuition", "campus",
                                       "learning", "curriculum"],
            },
            {
                "query": "energy oil gas renewable",
                "description": "Energy sector news",
                "relevance_keywords": ["energy", "oil", "gas", "renewable", "solar", "wind power",
                                       "nuclear", "electricity", "power plant", "fuel",
                                       "petroleum", "opec"],
            },
            {
                "query": "security cyber data privacy",
                "description": "Cybersecurity and privacy news",
                "relevance_keywords": ["cyber", "security", "hack", "data breach", "privacy",
                                       "encryption", "malware", "ransomware", "phishing",
                                       "firewall", "surveillance", "data protection"],
            },
        ]

        test_queries = []

        for tq_def in test_query_definitions:
            # Find ALL documents in the corpus that match the relevance keywords
            relevant_docs = set()
            for doc_id, doc_meta in index.documents.items():
                title = doc_meta.get("title", "").lower()
                content = doc_meta.get("original_content", "").lower()
                combined = title + " " + content

                for keyword in tq_def["relevance_keywords"]:
                    if keyword.lower() in combined:
                        relevant_docs.add(doc_id)
                        break  # One keyword match is enough

            # Only include queries that have at least 2 relevant documents
            if len(relevant_docs) >= 2:
                test_queries.append({
                    "query": tq_def["query"],
                    "description": tq_def["description"],
                    "relevant_docs": list(relevant_docs),
                    "relevance_keywords": tq_def["relevance_keywords"],
                })

        print(f"[+] Created {len(test_queries)} test queries with keyword-based relevance judgments")
        for tq in test_queries:
            print(f"    - \"{tq['query']}\": {len(tq['relevant_docs'])} relevant docs")
        return test_queries

    def precision_at_k(self, retrieved_ids, relevant_ids, k):
        """Precision@K: fraction of top-K results that are relevant."""
        retrieved_k = retrieved_ids[:k]
        relevant_count = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
        return relevant_count / k if k > 0 else 0

    def recall_at_k(self, retrieved_ids, relevant_ids, k):
        """Recall@K: fraction of relevant documents found in top-K."""
        retrieved_k = set(retrieved_ids[:k])
        relevant_found = len(retrieved_k & relevant_ids)
        return relevant_found / len(relevant_ids) if relevant_ids else 0

    def f1_score(self, precision, recall):
        """F1 Score: harmonic mean of precision and recall."""
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def average_precision(self, retrieved_ids, relevant_ids):
        """Average Precision for a single query."""
        hits = 0
        sum_precision = 0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                hits += 1
                sum_precision += hits / (i + 1)

        return sum_precision / len(relevant_ids) if relevant_ids else 0

    def dcg_at_k(self, retrieved_ids, relevant_ids, k):
        """Discounted Cumulative Gain at K."""
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = 1 if doc_id in relevant_ids else 0
            dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg

    def ndcg_at_k(self, retrieved_ids, relevant_ids, k):
        """Normalized DCG at K."""
        dcg = self.dcg_at_k(retrieved_ids, relevant_ids, k)
        # Ideal DCG: all relevant docs at top
        ideal_retrieved = list(relevant_ids)[:k]
        ideal_dcg = self.dcg_at_k(ideal_retrieved, relevant_ids, k)
        return dcg / ideal_dcg if ideal_dcg > 0 else 0

    def evaluate(self, test_queries, k_values=[5, 10]):
        """
        Run full evaluation on test queries.

        Args:
            test_queries: List of {query, relevant_docs} dicts
            k_values: List of K values for Precision@K, Recall@K

        Returns:
            Evaluation results dict
        """
        print("\n" + "=" * 60)
        print("  IR SYSTEM EVALUATION")
        print("=" * 60)

        all_results = {
            "per_query": [],
            "aggregate": {},
        }

        # Metrics accumulators
        metrics = defaultdict(list)

        for i, tq in enumerate(test_queries):
            query = tq["query"]
            relevant_ids = set(tq["relevant_docs"])

            # Search with BM25
            results = self.search_engine.search(query, mode="bm25", top_k=max(k_values))
            retrieved_ids = [r["doc_id"] for r in results]

            query_metrics = {
                "query": query,
                "description": tq.get("description", ""),
                "num_relevant": len(relevant_ids),
                "num_retrieved": len(retrieved_ids),
            }

            for k in k_values:
                p_k = self.precision_at_k(retrieved_ids, relevant_ids, k)
                r_k = self.recall_at_k(retrieved_ids, relevant_ids, k)
                f1 = self.f1_score(p_k, r_k)
                ndcg = self.ndcg_at_k(retrieved_ids, relevant_ids, k)

                query_metrics[f"P@{k}"] = round(p_k, 4)
                query_metrics[f"R@{k}"] = round(r_k, 4)
                query_metrics[f"F1@{k}"] = round(f1, 4)
                query_metrics[f"NDCG@{k}"] = round(ndcg, 4)

                metrics[f"P@{k}"].append(p_k)
                metrics[f"R@{k}"].append(r_k)
                metrics[f"F1@{k}"].append(f1)
                metrics[f"NDCG@{k}"].append(ndcg)

            ap = self.average_precision(retrieved_ids, relevant_ids)
            query_metrics["AP"] = round(ap, 4)
            metrics["AP"].append(ap)

            all_results["per_query"].append(query_metrics)

            print(f"\n  Query {i+1}: \"{query}\"")
            print(f"    Relevant: {len(relevant_ids)}, Retrieved: {len(retrieved_ids)}")
            
            # Show which retrieved docs are relevant/not relevant
            for j, r_id in enumerate(retrieved_ids[:5]):
                is_rel = "RELEVANT" if r_id in relevant_ids else "NOT RELEVANT"
                doc = self.search_engine.index.get_document(r_id)
                title = doc["title"][:50] if doc else "?"
                print(f"      #{j+1} [{is_rel}] {title}...")
            
            for k in k_values:
                print(f"    P@{k}={query_metrics[f'P@{k}']:.3f}  "
                      f"R@{k}={query_metrics[f'R@{k}']:.3f}  "
                      f"F1@{k}={query_metrics[f'F1@{k}']:.3f}  "
                      f"NDCG@{k}={query_metrics[f'NDCG@{k}']:.3f}")

        # Compute aggregate metrics
        print(f"\n{'-' * 60}")
        print("  AGGREGATE METRICS (averaged over all queries)")
        print(f"{'-' * 60}")

        for metric_name, values in sorted(metrics.items()):
            avg_val = sum(values) / len(values) if values else 0
            all_results["aggregate"][f"Mean_{metric_name}"] = round(avg_val, 4)
            print(f"    Mean {metric_name}: {avg_val:.4f}")

        # MAP
        map_score = sum(metrics["AP"]) / len(metrics["AP"]) if metrics["AP"] else 0
        all_results["aggregate"]["MAP"] = round(map_score, 4)
        print(f"    MAP:      {map_score:.4f}")
        print("=" * 60)

        self.results = all_results
        return all_results

    def save_evaluation(self, filepath="data/evaluation_results.json"):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"[+] Evaluation results saved to {filepath}")

    def generate_report(self):
        """Generate a formatted evaluation report string."""
        if not self.results:
            return "No evaluation results available."

        report = []
        report.append("=" * 60)
        report.append("  IR SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"\n  Number of test queries: {len(self.results['per_query'])}")
        report.append("\n  Aggregate Metrics:")
        for metric, value in self.results["aggregate"].items():
            report.append(f"    {metric}: {value:.4f}")

        report.append(f"\n{'-' * 60}")
        report.append("  Per-Query Results:")
        for pq in self.results["per_query"]:
            report.append(f"\n  Query: \"{pq['query']}\"")
            report.append(f"    Description: {pq.get('description', '')}")
            for key, val in pq.items():
                if key not in ["query", "description"]:
                    report.append(f"    {key}: {val}")

        return "\n".join(report)
