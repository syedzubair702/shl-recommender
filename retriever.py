"""
Retriever: semantic search over SHL catalog.
Primary:  FAISS + sentence-transformers (when available)
Fallback: TF-IDF BM25-style keyword search (always available, no extra deps)
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional
from functools import lru_cache

DATA_DIR = Path(__file__).parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_PATH = DATA_DIR / "catalog_index.faiss"
META_PATH = DATA_DIR / "catalog_index_meta.json"


@lru_cache(maxsize=1)
def load_catalog() -> list:
    return json.loads(CATALOG_PATH.read_text())


@lru_cache(maxsize=1)
def _load_faiss():
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        if not INDEX_PATH.exists():
            return None, None, None
        index = faiss.read_index(str(INDEX_PATH))
        meta = json.loads(META_PATH.read_text())
        model = SentenceTransformer(meta["model"])
        print(f"[Retriever] FAISS loaded: {meta['n']} items")
        return index, model, meta
    except Exception as e:
        print(f"[Retriever] FAISS unavailable ({type(e).__name__}), using TF-IDF.")
        return None, None, None


def _tokenize(text: str) -> list:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def build_document(item: dict) -> str:
    parts = [item["name"]]
    if item.get("description"):
        parts.append(item["description"])
    if item.get("test_type_labels"):
        parts.append(" ".join(item["test_type_labels"]))
    if item.get("test_types"):
        parts.append(" ".join(item["test_types"]))

    name_lower = item["name"].lower()
    desc_lower = item.get("description", "").lower()
    combined = name_lower + " " + desc_lower

    expansions = {
        "java": "software developer programming coding technical backend",
        "python": "software developer data science scripting analytics",
        "javascript": "frontend web developer software engineer",
        "sql": "database data analyst backend engineering",
        "c++": "systems software developer engineering",
        "numerical": "quantitative analytical finance math data numbers statistics",
        "verbal": "communication language reading comprehension writing",
        "personality": "behaviour traits culture fit leadership team interpersonal",
        "opq": "personality occupational behaviour leadership selection development",
        "sales": "commercial revenue customer client business development",
        "customer": "service client contact center support retail",
        "leadership": "management senior executive director manager",
        "graduate": "entry level junior early career trainee",
        "machine learning": "ml ai data scientist artificial intelligence",
        "devops": "cloud infrastructure platform engineering",
        "aws": "cloud amazon web services infrastructure platform",
        "cognitive": "ability aptitude reasoning intelligence general",
    }
    for kw, expansion in expansions.items():
        if kw in combined:
            parts.append(expansion)

    return " ".join(parts)


@lru_cache(maxsize=1)
def _build_tfidf_index():
    catalog = load_catalog()
    doc_texts = [build_document(item) for item in catalog]
    doc_tokens = [_tokenize(t) for t in doc_texts]

    N = len(catalog)
    df = defaultdict(int)
    for tokens in doc_tokens:
        for tok in set(tokens):
            df[tok] += 1
    idf = {tok: math.log((N + 1) / (count + 1)) + 1 for tok, count in df.items()}

    return idf, doc_tokens, catalog


def tfidf_search(query: str, top_k: int = 10) -> list:
    idf, doc_tokens, catalog = _build_tfidf_index()
    query_tokens = _tokenize(query)

    if not query_tokens:
        return list(catalog[:top_k])

    scores = []
    for i, tokens in enumerate(doc_tokens):
        tf = defaultdict(float)
        for tok in tokens:
            tf[tok] += 1
        doc_len = len(tokens)

        score = 0.0
        for qtok in query_tokens:
            if qtok in tf:
                k1, b, avgdl = 1.5, 0.75, 100
                freq = tf[qtok]
                score += idf.get(qtok, 1.0) * (
                    (freq * (k1 + 1)) /
                    (freq + k1 * (1 - b + b * doc_len / avgdl))
                )

        name_lower = catalog[i]["name"].lower()
        for qtok in query_tokens:
            if qtok in name_lower:
                score += 2.0

        scores.append((score, i))

    scores.sort(key=lambda x: -x[0])
    top = [catalog[idx].copy() for _, idx in scores[:top_k] if scores[0][0] > 0]
    return top if top else list(catalog[:top_k])


def semantic_search(query: str, top_k: int = 10) -> list:
    index, model, meta = _load_faiss()
    catalog = load_catalog()

    if index is not None and model is not None:
        import faiss
        import numpy as np
        vec = model.encode([query], show_progress_bar=False)
        vec = np.array(vec, dtype=np.float32)
        faiss.normalize_L2(vec)
        scores, indices = index.search(vec, min(top_k, len(catalog)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = catalog[idx].copy()
                item["_score"] = float(score)
                results.append(item)
        return results

    return tfidf_search(query, top_k)


def get_assessment_by_name(name: str) -> Optional[dict]:
    catalog = load_catalog()
    name_lower = name.lower().strip()
    for item in catalog:
        if item["name"].lower() == name_lower:
            return item
    for item in catalog:
        if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
            return item
    return None


def retrieve(query: str, top_k: int = 10, filters: Optional[dict] = None) -> list:
    candidate_k = min(top_k * 3, 30) if filters else top_k
    results = semantic_search(query, top_k=candidate_k)

    if filters:
        filtered = []
        for item in results:
            if filters.get("remote_only") and not item.get("remote_testing"):
                continue
            if filters.get("test_types"):
                if not any(t in item.get("test_types", []) for t in filters["test_types"]):
                    continue
            filtered.append(item)
        results = filtered

    return results[:top_k]
