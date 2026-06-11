import os
import string
from typing import Any, Dict, List, Tuple

import chromadb
import nltk
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

os.environ["HF_HOME"] = r"E:/models/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"E:/models/sentence_transformers"
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file

CHROMA_DIR = r"E:/Rag_Project/chroma_db"
COLLECTION_NAME = "fedadapriv"

# ── load once, reuse everywhere ──────────────────────────────────────────────
def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=r"E:/models/sentence_transformers"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    all_data = vectorstore._collection.get(include=["documents", "metadatas"])
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25_index = BM25Okapi(tokenized_corpus)

    return vectorstore, bm25_index, documents, metadatas


# ── tokenizer ────────────────────────────────────────────────────────────────
def tokenize(text):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    punctuation = set(string.punctuation)

    text = text.lower()
    tokens = nltk.word_tokenize(text)
    # keep only real words — NOT punctuation, NOT stopwords
    tokens = [t for t in tokens if t not in punctuation and t not in stopwords]
    return tokens


# ── semantic search ───────────────────────────────────────────────────────────
def semantic_search(query, vectorstore, k=10):
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [
        {"content": doc.page_content, "metadata": doc.metadata, "score": score}
        for doc, score in results
    ]


# ── bm25 search ───────────────────────────────────────────────────────────────
def bm25_search(query, documents, metadatas, bm25_index, k=10):
    scores = bm25_index.get_scores(tokenize(query))
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"content": documents[idx], "metadata": metadatas[idx], "score": scores[idx]}
        for idx in top_k
    ]


# ── reciprocal rank fusion ────────────────────────────────────────────────────
def reciprocal_rank_fusion(semantic_results, bm25_results, k=60):
    scores = {}

    for rank, result in enumerate(semantic_results):
        content = result["content"]
        if content not in scores:
            scores[content] = {"score": 0.0, "metadata": result["metadata"]}
        scores[content]["score"] += 1 / (k + rank + 1)

    for rank, result in enumerate(bm25_results):
        content = result["content"]
        if content not in scores:
            scores[content] = {"score": 0.0, "metadata": result["metadata"]}
        scores[content]["score"] += 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)


# ── main retrieve function ────────────────────────────────────────────────────
def retrieve(query, vectorstore, bm25_index, documents, metadatas, top_k=5, k=None):
    if k is not None:
        top_k = k

    semantic_results = semantic_search(query, vectorstore, k=10)
    bm25_results = bm25_search(query, documents, metadatas, bm25_index, k=10)
    fused = reciprocal_rank_fusion(semantic_results, bm25_results)

    return [
        {"content": content, "metadata": data["metadata"]}
        for content, data in fused[:top_k]
    ]



# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vectorstore, bm25_index, documents, metadatas = load_retriever()
    print(f"Chunks loaded: {len(documents)}")
    print(f"Sample chunk: {documents[0][:200] if documents else 'EMPTY'}")

    query = "What is the epsilon used?"
    results = retrieve(query, vectorstore, bm25_index, documents, metadatas, top_k=5)

    print(f"\nQuery: {query}")
    print(f"Top {len(results)} chunks:\n")
    for i, chunk in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(f"Source: {chunk['metadata'].get('source', '?')} | Page: {chunk['metadata'].get('page', '?')}")
        print(chunk["content"][:300])
        print()