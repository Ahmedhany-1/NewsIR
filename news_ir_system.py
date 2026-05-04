"""
News IR System - Main Orchestrator
====================================
A complete Information Retrieval system for news articles.

Purpose: Search engine for real-world news articles collected from
         major international news sources (BBC, Reuters, CNN, NPR, etc.)
         
Pipeline:
    1. Data Collection   -> RSS feeds + web scraping
    2. Preprocessing     -> Tokenize, normalize, stem, lemmatize, remove stop words
    3. Indexing          -> Build inverted index with TF-IDF and BM25
    4. Search & Retrieve -> Ranked retrieval with multiple scoring methods
    5. Evaluation        -> Precision, Recall, F1, MAP, NDCG
    6. Web Interface     -> Flask-based search UI
"""

import os
import sys
import json
import time

from data_collector import collect_from_rss, save_articles, load_articles
from preprocessor import TextPreprocessor, save_processed, load_processed
from indexer import InvertedIndex
from search_engine import SearchEngine
from evaluator import IREvaluator


def run_full_pipeline(max_per_feed=15, fetch_full_text=True):
    """
    Run the complete IR pipeline from data collection to evaluation.
    """
    start_time = time.time()

    print("\n" + "#" * 60)
    print("  NEWS ARTICLE INFORMATION RETRIEVAL SYSTEM")
    print("  " + "-" * 56)
    print("  A complete IR system for searching news articles")
    print("  from real-world international news sources.")
    print("#" * 60)

    # ------------------------------------------------
    # STEP 1: DATA COLLECTION
    # ------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  STEP 1: DATA COLLECTION")
    print("=" * 60)

    raw_path = "data/raw_articles.json"
    if os.path.exists(raw_path):
        print(f"[*] Found existing data at {raw_path}")
        articles = load_articles(raw_path)
        if len(articles) < 20:
            print("[*] Too few articles, re-collecting...")
            articles = collect_from_rss(max_per_feed=max_per_feed, fetch_full_text=fetch_full_text)
            save_articles(articles, raw_path)
    else:
        articles = collect_from_rss(max_per_feed=max_per_feed, fetch_full_text=fetch_full_text)
        save_articles(articles, raw_path)

    if not articles:
        print("[ERROR] No articles collected. Check your internet connection.")
        sys.exit(1)

    # ------------------------------------------------
    # STEP 2: DATA PREPROCESSING
    # ------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 60)

    preprocessor = TextPreprocessor(
        use_stemming=True,
        use_lemmatization=True,
        remove_stopwords=True,
    )
    processed_docs = preprocessor.preprocess_documents(articles)
    save_processed(processed_docs, "data/processed_articles.json")

    # Vocabulary statistics
    vocab_stats = preprocessor.get_vocabulary_stats(processed_docs)
    print(f"\n  Vocabulary Statistics:")
    print(f"  - Vocabulary size:  {vocab_stats['vocabulary_size']}")
    print(f"  - Total tokens:     {vocab_stats['total_tokens']}")
    print(f"  - Avg frequency:    {vocab_stats['avg_frequency']}")
    print(f"  - Top 10 terms:     {[t[0] for t in vocab_stats['most_common_30'][:10]]}")

    # ------------------------------------------------
    # STEP 3: BUILD INVERTED INDEX
    # ------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  STEP 3: BUILD INVERTED INDEX")
    print("=" * 60)

    index = InvertedIndex()
    index.build_index(processed_docs)
    index.save_index("data/inverted_index.pkl")

    # ------------------------------------------------
    # STEP 4: SEARCH & RETRIEVAL TEST
    # ------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  STEP 4: SEARCH & RETRIEVAL DEMO")
    print("=" * 60)

    engine = SearchEngine(index, preprocessor)

    demo_queries = [
        "artificial intelligence technology",
        "climate change global warming",
        "economic policy government",
        "health pandemic virus",
        "war conflict peace",
    ]

    for query in demo_queries:
        results = engine.search(query, mode="bm25", top_k=5)
        engine.display_results(results, query)

    # ------------------------------------------------
    # STEP 5: EVALUATION
    # ------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  STEP 5: EVALUATION & QUALITY CHECK")
    print("=" * 60)

    evaluator = IREvaluator(engine)
    test_queries = evaluator.create_test_queries(index)
    
    if test_queries:
        eval_results = evaluator.evaluate(test_queries, k_values=[5, 10])
        evaluator.save_evaluation("data/evaluation_results.json")
    else:
        print("[WARN] Not enough data for meaningful evaluation.")
        eval_results = {}

    # ------------------------------------------------
    # SUMMARY
    # ------------------------------------------------
    elapsed = time.time() - start_time
    idx_stats = index.get_index_stats()

    print("\n\n" + "#" * 60)
    print("  PIPELINE COMPLETE - SUMMARY")
    print("#" * 60)
    print(f"  Total time:           {elapsed:.1f} seconds")
    print(f"  Articles collected:   {len(articles)}")
    print(f"  Documents indexed:    {idx_stats['total_documents']}")
    print(f"  Vocabulary size:      {idx_stats['vocabulary_size']}")
    print(f"  Sources:              {', '.join(idx_stats['sources'])}")
    print(f"  Index saved to:       data/inverted_index.pkl")
    if eval_results and "aggregate" in eval_results:
        agg = eval_results["aggregate"]
        print(f"  MAP Score:            {agg.get('MAP', 'N/A')}")
        print(f"  Mean P@5:             {agg.get('Mean_P@5', 'N/A')}")
        print(f"  Mean F1@5:            {agg.get('Mean_F1@5', 'N/A')}")
    print(f"\n  Run 'python app.py' to start the web search interface!")
    print("#" * 60)

    return index, engine, preprocessor


def interactive_search():
    """Run an interactive command-line search session."""
    index = InvertedIndex()
    if not index.load_index("data/inverted_index.pkl"):
        print("[!] No index found. Run the full pipeline first.")
        print("    Execute: python news_ir_system.py --build")
        return

    preprocessor = TextPreprocessor()
    engine = SearchEngine(index, preprocessor)

    print("\n" + "=" * 60)
    print("  NEWS SEARCH ENGINE - Interactive Mode")
    print("  Type your query and press Enter. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        query = input("\n  Search> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        results = engine.search(query, mode="bm25", top_k=10)
        engine.display_results(results, query)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="News Article IR System")
    parser.add_argument("--build", action="store_true", help="Run full pipeline (collect, preprocess, index, evaluate)")
    parser.add_argument("--search", action="store_true", help="Interactive search mode")
    parser.add_argument("--max-articles", type=int, default=15, help="Max articles per RSS feed")
    parser.add_argument("--no-fulltext", action="store_true", help="Skip full-text fetching (faster)")

    args = parser.parse_args()

    if args.search:
        interactive_search()
    elif args.build:
        run_full_pipeline(
            max_per_feed=args.max_articles,
            fetch_full_text=not args.no_fulltext,
        )
    else:
        # Default: run full pipeline
        run_full_pipeline(
            max_per_feed=args.max_articles,
            fetch_full_text=not args.no_fulltext,
        )
