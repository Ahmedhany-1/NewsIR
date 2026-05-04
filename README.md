# NewsIR — News Article Information Retrieval System

A complete Information Retrieval system that collects, indexes, and searches real-world news articles from major international sources using BM25 and TF-IDF ranking.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![NLTK](https://img.shields.io/badge/NLTK-3.9-orange)

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ahmedfathyhany/NewsIR.git
cd NewsIR
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 4. Build the system (collect articles, preprocess, index, evaluate)
```bash
python news_ir_system.py --build
```
This will:
- Collect ~80 news articles from RSS feeds (BBC, CNN, NPR, The Guardian, etc.)
- Preprocess text (tokenize, normalize, stem, lemmatize, remove stop words)
- Build an inverted index with BM25 and TF-IDF scoring
- Run evaluation with Precision, Recall, F1, MAP, NDCG metrics

### 5. Start the web search interface
```bash
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

### 6. (Optional) Interactive CLI search
```bash
python news_ir_system.py --search
```

---

## Project Overview

### Purpose & Scope
NewsIR is a **news search engine** that enables users to search through real-world news articles using intelligent ranking algorithms. It demonstrates a full IR pipeline from data collection to evaluation.

### Data Sources
Articles are collected from real-world sources via **RSS feeds** and **web scraping**:

| Source | Category |
|--------|----------|
| BBC News | General |
| BBC Technology | Technology |
| BBC Science | Science |
| Reuters | General / Tech |
| NPR News | General |
| Al Jazeera | General |
| The Guardian | World / Tech |
| CNN Top Stories | General |

---

## IR Pipeline

### 1. Data Collection (`data_collector.py`)
- Parses RSS feeds using `feedparser`
- Fetches full article text via `requests` + `BeautifulSoup`
- Extracts content from semantic HTML elements
- Stores raw articles in JSON format

### 2. Preprocessing (`preprocessor.py`)
| Step | Method | Tool |
|------|--------|------|
| Text Normalization | Lowercase, remove URLs/HTML/special chars | Regex |
| Tokenization | Word boundary detection | NLTK `word_tokenize` |
| Stop Word Removal | English stopwords + custom news terms | NLTK stopwords |
| Lemmatization | Dictionary base form | NLTK WordNet |
| Stemming | Root form reduction | NLTK Porter Stemmer |

### 3. Indexing (`indexer.py`)
- **Inverted Index**: `term → {doc_id: frequency, ...}`
- **TF-IDF**: Log-normalized TF × IDF scoring
- **BM25 (Okapi)**: `k1=1.5, b=0.75` with length normalization
- **Boolean Retrieval**: AND/OR set operations on posting lists

### 4. Search & Retrieval (`search_engine.py`)
- Query preprocessing (same pipeline as documents)
- Ranked retrieval with BM25, TF-IDF, or Boolean modes
- **Keyword highlighting** in titles and snippets using `<mark>` tags
- Snippet generation from most relevant sentences

### 5. Evaluation (`evaluator.py`)
Uses **keyword-based relevance judgments** (not pseudo-relevance) for honest metrics:

| Metric | Value |
|--------|-------|
| MAP | 0.342 |
| Mean Precision@5 | 0.820 |
| Mean Recall@5 | 0.223 |
| Mean F1@5 | 0.342 |
| Mean NDCG@5 | 0.842 |

### 6. Web Interface (`app.py` + `templates/index.html`)
- Clean editorial design with warm color palette
- Three search modes: BM25, TF-IDF, Boolean AND
- Keyword highlighting in results
- System statistics and evaluation metrics on home page
- REST API at `/api/search?q=your+query`

---

## Project Structure

```
NewsIR/
├── news_ir_system.py      # Main pipeline orchestrator
├── data_collector.py      # RSS feed collection & web scraping
├── preprocessor.py        # Text preprocessing (tokenize, stem, lemmatize)
├── indexer.py             # Inverted index with TF-IDF & BM25
├── search_engine.py       # Query processing & ranked retrieval
├── evaluator.py           # Evaluation (Precision, Recall, F1, MAP, NDCG)
├── app.py                 # Flask web application
├── templates/
│   └── index.html         # Web search interface
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## Command Reference

```bash
# Full pipeline: collect → preprocess → index → evaluate
python news_ir_system.py --build

# Full pipeline with more articles per feed
python news_ir_system.py --build --max-articles 20

# Faster build (skip full-text scraping, use RSS summaries only)
python news_ir_system.py --build --no-fulltext

# Interactive CLI search
python news_ir_system.py --search

# Start web interface
python app.py
```

---

## Technologies

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| NLP | NLTK (tokenization, stemming, lemmatization) |
| Web Scraping | BeautifulSoup4, Requests, Feedparser |
| Indexing | Custom Inverted Index |
| Ranking | BM25 (Okapi), TF-IDF |
| Web Framework | Flask |
| Evaluation | Precision, Recall, F1, MAP, NDCG |
