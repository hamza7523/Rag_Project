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

os.environ["HF_HOME"] = r"E:/models/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"E:/models/sentence_transformers"

CHROMA_DIR = r"E:/Rag_Project/chroma_db"
COLLECTION_NAME = "fedadapriv"

def load_retriever():
    
    # build bm25 index from chunks in chromadb
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=r"E:/models/sentence_transformers"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Pull all chunks out of ChromaDB to build BM25 index
    all_data = vectorstore._collection.get(include=["documents", "metadatas"])
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    # Tokenize for BM25
    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    return vectorstore, bm25_index, documents, metadatas
    
def semantic_search(query, vectorstore, k=10):
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    formatted = []
    for doc, score in results:
        formatted.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })
    
    return formatted
def bm25_search(query, chunks, bm25_index, k=5): 
    # returns top-k chunks from bm25
    bm25_results = bm25_index.get_scores(tokenize(query))
    top_k = sorted(range(len(bm25_results)),key= lambda i: bm25_results[i],reverse=True)[:k]
    results = []
    for idx in top_k: # type: ignore
        results.append({
            "content": documents[idx],
            "metadata": metadatas[idx],
            "score": bm25_results[idx]
        })
    return results
    



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
    
    reranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    return reranked


def retrieve(query, top_k=5):
    vectorstore, bm25_index, documents, metadatas = load_retriever()
    print(bm25_index)
    
    semantic_results = semantic_search(query, vectorstore, k=10)
    bm25_results = bm25_search(query, documents, bm25_index, k=10)
    
    fused = reciprocal_rank_fusion(semantic_results, bm25_results)
    
    top_chunks = []
    for content, data in fused[:top_k]:
        top_chunks.append({
            "content": content,
            "metadata": data["metadata"]
        })
    
    return top_chunks
def tokenize(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token in string.punctuation and token not in nltk.corpus.stopwords.words('english')]
    return tokens

if __name__ == "__main__":
    # Debug first
    vectorstore, bm25_index, documents, metadatas = load_retriever()
    print(f"Chunks loaded: {len(documents)}")
    print(f"Sample chunk: {documents[0][:200] if documents else 'EMPTY'}")
    query = "What is the epsilon used?"
    results = retrieve(query)
    
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} chunks:\n")
    for i, chunk in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(f"Source: {chunk['metadata'].get('source', '?')} | Page: {chunk['metadata'].get('page', '?')}")
        print(chunk['content'][:300])
        print()