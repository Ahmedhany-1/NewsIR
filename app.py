"""
Flask Web Application for News IR System
==========================================
Beautiful web-based search interface for the News IR System.
"""

import os
import json
from flask import Flask, render_template, request, jsonify

from preprocessor import TextPreprocessor
from indexer import InvertedIndex
from search_engine import SearchEngine

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global objects
index = InvertedIndex()
preprocessor = TextPreprocessor()
engine = None


def init_engine():
    """Initialize the search engine with saved index."""
    global engine
    if index.load_index("data/inverted_index.pkl"):
        engine = SearchEngine(index, preprocessor)
        print("[+] Search engine ready!")
    else:
        print("[!] No index found. Run 'python news_ir_system.py --build' first.")


@app.route("/")
def home():
    """Render the search home page."""
    stats = index.get_index_stats() if index.total_docs > 0 else {}
    
    # Load evaluation results if available
    eval_results = {}
    eval_path = "data/evaluation_results.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_results = json.load(f)
    
    return render_template("index.html", stats=stats, eval_results=eval_results)


@app.route("/search")
def search():
    """Execute a search query and return results."""
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "bm25")
    top_k = int(request.args.get("top_k", 10))

    if not query or not engine:
        return render_template("index.html", query=query, results=[], stats=index.get_index_stats(), eval_results={})

    results = engine.search(query, mode=mode, top_k=top_k)
    stats = index.get_index_stats()
    
    return render_template(
        "index.html",
        query=query,
        results=results,
        mode=mode,
        stats=stats,
        eval_results={},
    )


@app.route("/api/search")
def api_search():
    """REST API endpoint for search."""
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "bm25")
    top_k = int(request.args.get("top_k", 10))

    if not query or not engine:
        return jsonify({"error": "No query provided or engine not initialized", "results": []})

    results = engine.search(query, mode=mode, top_k=top_k)
    return jsonify({"query": query, "mode": mode, "count": len(results), "results": results})


@app.route("/stats")
def stats_page():
    """Show index and evaluation statistics."""
    stats = index.get_index_stats() if index.total_docs > 0 else {}
    eval_results = {}
    eval_path = "data/evaluation_results.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_results = json.load(f)
    return render_template("index.html", stats=stats, eval_results=eval_results, show_stats=True)


if __name__ == "__main__":
    init_engine()
    print("\n[*] Starting web server at http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
