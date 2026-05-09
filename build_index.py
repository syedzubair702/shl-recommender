"""
Build FAISS vector index over SHL catalog for semantic retrieval.
Outputs: data/catalog_index.faiss + data/catalog_index_meta.json
"""

import json
import sys
import numpy as np
from pathlib import Path

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "faiss-cpu", "sentence-transformers", "--quiet"
    ], check=True)
    import faiss
    from sentence_transformers import SentenceTransformer


DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_PATH = DATA_DIR / "catalog_index.faiss"
META_PATH = DATA_DIR / "catalog_index_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast, 384-dim


def build_document(item: dict) -> str:
    """Build rich text document for embedding from catalog item."""
    parts = [item["name"]]

    if item.get("description"):
        parts.append(item["description"])

    if item.get("test_type_labels"):
        parts.append("Test types: " + ", ".join(item["test_type_labels"]))

    if item.get("test_types"):
        parts.append("Categories: " + ", ".join(item["test_types"]))

    if item.get("duration_minutes"):
        parts.append(f"Duration: {item['duration_minutes']} minutes")

    if item.get("languages"):
        parts.append("Languages: " + ", ".join(item["languages"]))

    # Add role keywords derived from name/description for better retrieval
    name_lower = item["name"].lower()
    if any(kw in name_lower for kw in ["java", "python", "javascript", "sql", "c++", "c#"]):
        parts.append("software developer programming coding technical role")
    if any(kw in name_lower for kw in ["numerical", "number", "math", "calculation"]):
        parts.append("quantitative analytical finance data")
    if any(kw in name_lower for kw in ["verbal", "language", "english", "communication"]):
        parts.append("communication writing language comprehension")
    if any(kw in name_lower for kw in ["personality", "opq", "behaviour", "style"]):
        parts.append("personality traits behaviour culture fit leadership team")
    if any(kw in name_lower for kw in ["sales", "customer", "contact", "retail"]):
        parts.append("sales customer service client-facing commercial")
    if any(kw in name_lower for kw in ["leadership", "management", "managerial", "executive"]):
        parts.append("leadership management senior executive director")
    if any(kw in name_lower for kw in ["graduate", "entry", "early"]):
        parts.append("graduate entry level junior early career")
    if any(kw in name_lower for kw in ["machine learning", "data science", "ml", "ai"]):
        parts.append("machine learning data science artificial intelligence analytics")
    if any(kw in name_lower for kw in ["devops", "aws", "cloud", "agile"]):
        parts.append("devops cloud infrastructure platform engineering")

    return " | ".join(parts)


def build_index():
    catalog = json.loads(CATALOG_PATH.read_text())
    print(f"Building index for {len(catalog)} assessments...")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    documents = [build_document(item) for item in catalog]

    print("Encoding documents...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build flat inner product index (cosine sim after normalization)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps({
        "model": MODEL_NAME,
        "dim": dim,
        "n": len(catalog),
        "documents": documents,
    }, indent=2))

    print(f"Index saved to {INDEX_PATH}")
    print(f"Meta saved to {META_PATH}")


if __name__ == "__main__":
    build_index()
