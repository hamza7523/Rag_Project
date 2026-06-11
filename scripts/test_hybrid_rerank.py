import sys
from pathlib import Path

# Ensure repository root is on sys.path so retrieval_utils imports correctly.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from retrieval_utils import hybrid_rerank_candidates


def run_test():
    semantic_results = [
        {"content": "Total revenue: $4,410,000\nUnits sold: 2,100,000", "metadata": {"source": "Dew_sales.txt"}, "score": 0.9},
        {"content": "We estimate revenue around $4.4M", "metadata": {"source": "Dew_notes.txt"}, "score": 0.6},
        {"content": "No revenue mentioned here", "metadata": {"source": "Other.txt"}, "score": 0.2},
    ]

    query = "What is the total revenue $4,410,000?"
    results = hybrid_rerank_candidates(semantic_results, query, metadata_filter={"source": "Dew"}, top_k=3)
    print("Hybrid rerank results:")
    for r in results:
        print(r["metadata"]["source"], "->", r["content"][:80].replace("\n", " "))


if __name__ == '__main__':
    run_test()
