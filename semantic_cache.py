# semantic_cache.py

import numpy as np
from sentence_transformers import SentenceTransformer
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
import os

# ── Constants ──────────────────────────────────────────────────────────────────

# The similarity threshold — queries with cosine similarity above this
# are considered "same enough" to return the cached answer.
# 0.90 is a safe starting point; tune down if too few hits, up if wrong answers returned.
SIMILARITY_THRESHOLD = 0.80

# How long (seconds) a cache entry lives before auto-expiring.
# 3600 = 1 hour. In production, align this with your ingestion frequency.
# If new docs are ingested every 6 hours, TTL should be < 6 hours to avoid stale answers.
CACHE_TTL = 3600

# Dimension of all-MiniLM-L6-v2 embeddings.
# Must match exactly — Redis uses this to build the HNSW index structure.
EMBEDDING_DIM = 384

# Name of the Redis index we'll create. Acts like a "table name" for our cache.
INDEX_NAME = "semantic_cache"


class SemanticCache:

    def __init__(self):
        # Load the same embedding model used in retriever.py
        # CRITICAL: must be the same model — cache lookup compares embeddings,
        # so mixing models produces nonsensical similarity scores.
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Define the schema for our Redis index.
        # Each cached entry has:
        #   - "query_embedding": the vector (HNSW index for fast ANN search)
        #   - "answer": the generated text we want to return on a hit
        self.schema = {
            "index": {
                "name": INDEX_NAME,
                "prefix": "cache:",       # All keys stored as cache:<uuid>
                "storage_type": "hash",   # Redis HASH type — simple field:value store
            },
            "fields": [
                {
                    "name": "query_embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",      # Hierarchical Navigable Small World graph
                        "dims": EMBEDDING_DIM,    # Must match your encoder output
                        "distance_metric": "cosine",  # Cosine similarity for semantic matching
                        "datatype": "float32",
                    }
                },
                {
                    "name": "answer",
                    "type": "text",   # Plain text field — no indexing needed, just storage
                }
            ]
        }

        # Create the SearchIndex object using our schema.
        # redis_url points to your local Docker container.
        self.index = SearchIndex.from_dict(self.schema)
        self.index.connect("redis://localhost:6379")

        # Create the index in Redis if it doesn't already exist.
        # overwrite=False means: skip silently if index is already there.
        # This is safe to call every startup — idempotent.
        self.index.create(overwrite=False)

        # Keep a direct redis client reference for TTL operations.
        # SearchIndex doesn't expose TTL setting, so we handle it manually.
        self.redis_client = self.index.client

    def _embed(self, text: str) -> np.ndarray:
        # Encode the query into a float32 numpy array.
        # normalize_embeddings=True ensures vectors are unit-length —
        # required for cosine similarity to work correctly (dot product = cosine when normalized).
        return self.encoder.encode(text, normalize_embeddings=True).astype(np.float32)

    def get(self, query: str) -> str | None:
        # Embed the incoming query
        query_vector = self._embed(query)

        # Build a KNN vector query:
        # "Find the 1 most similar cached query to this vector"
        # return_fields: which Redis hash fields to include in results
        # num_results=1: we only need the closest match
        vector_query = VectorQuery(
            vector=query_vector.tolist(),
            vector_field_name="query_embedding",
            return_fields=["answer", "vector_distance"],
            num_results=1,
        )

        results = self.index.query(vector_query)

        # If no results at all, it's a clean miss
        if not results:
            return None

        best = results[0]

        # vector_distance is cosine DISTANCE (0 = identical, 1 = opposite)
        # Convert to similarity: similarity = 1 - distance
        similarity = 1 - float(best["vector_distance"])

        # Only return the cached answer if similarity clears our threshold
        if similarity >= SIMILARITY_THRESHOLD:
            print(f"[Cache HIT] similarity={similarity:.3f}")
            return best["answer"]

        print(f"[Cache MISS] best similarity={similarity:.3f} (threshold={SIMILARITY_THRESHOLD})")
        return None

    def set(self, query: str, answer: str) -> None:
        import uuid

        # Embed the query for storage
        query_vector = self._embed(query)

        # Generate a unique key for this cache entry
        key = f"cache:{uuid.uuid4().hex}"

        # Store as a Redis HASH with two fields: the vector and the answer text
        # tobytes() converts the numpy array to raw bytes — how Redis stores vectors
        self.redis_client.hset(key, mapping={
            "query_embedding": query_vector.tobytes(),
            "answer": answer,
        })

        # Set the TTL — after CACHE_TTL seconds Redis auto-deletes this key
        # This prevents stale answers from living forever after document updates
        self.redis_client.expire(key, CACHE_TTL)

        print(f"[Cache SET] key={key}")

    def close(self) -> None:
        # Cleanly close the Redis connection
        # Should be called in FastAPI's lifespan shutdown
        self.redis_client.close()