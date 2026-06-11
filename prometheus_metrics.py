from prometheus_client import Counter, Histogram, Gauge  # ← add Gauge

Request_Counter = Counter(
    'rag_api_requests_total',
    'Total number of requests to the RAG API',
    ["status"]
)

Request_Latency = Histogram(
    'rag_api_request_latency_seconds',
    'Latency of RAG API requests in seconds',
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

Active_Requests = Gauge(  # ← was Counter, must be Gauge
    'rag_api_active_requests',
    'Number of active requests to the RAG API'
)

CACHE_HIT_COUNTER = Counter(
    "rag_cache_hits_total",
    "Total number of requests served from semantic cache",
)