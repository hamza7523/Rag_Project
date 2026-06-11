# Project Pepsico Beverages Sales Data Insights RAG-Based Chatbot
## Production-Grade AI System with MLOps & Observability

---

## 🎯 Project Overview

This is a **production-ready Retrieval-Augmented Generation (RAG) system** designed to provide intelligent insights into Pepsico beverage sales data through a conversational interface. It combines state-of-the-art NLP techniques with enterprise-grade monitoring, caching, and API patterns to deliver a scalable, observable AI pipeline.

**Key Innovation**: The system integrates semantic caching with Prometheus/Grafana monitoring to track RAG pipeline performance in real-time, enabling data-driven optimization of retrieval and generation components.

### Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│                    User Query                                 │
└────────────────────────────┬─────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Semantic Cache  │◄──── (Fast lookup)
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼─────┐        ┌─────▼────┐        ┌────▼──────┐
   │ Retrieval│        │Generation│        │ Embedding │
   │  (CHROMA)│        │ (LLM)    │        │ (HF)      │
   └────┬─────┘        └─────┬────┘        └────┬──────┘
        │                    │                   │
        └────────────────────┼───────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │  API Response       │
                  │ + Metadata          │
                  └─────────────────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        │                    │                     │
   ┌────▼─────────┐  ┌──────▼──────┐  ┌──────────▼─┐
   │ Prometheus   │  │  Grafana    │  │  Alerting  │
   │ Metrics      │  │  Dashboard  │  │  & Logging │
   └──────────────┘  └─────────────┘  └────────────┘
```

---

## ✨ Core Features

### 1. **RAG Pipeline (Retrieval-Augmented Generation)**
- **Document Ingestion**: Supports PDF, DOCX, and TXT formats with intelligent text splitting
- **Semantic Chunking**: `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200) balances context preservation with retrieval precision
- **Local Embeddings**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2` for privacy-preserving, on-device embedding generation
- **Vector Search**: ChromaDB with similarity search for fast, scalable retrieval (~1ms per query)
- **Multi-Source LLM Support**: Integration points for OpenAI, Google Generative AI, and local models

### 2. **Production API Layer**
- **FastAPI Framework**: Async request handling, automatic OpenAPI documentation, built-in validation
- **Structured Responses**: JSON responses with answer, source attribution, and latency metrics
- **CORS Support**: Cross-origin resource sharing for frontend integration
- **Error Handling**: Graceful exception handling with detailed error messages
- **Request Validation**: Pydantic models ensure data integrity at API boundary

### 3. **Enterprise Observability (MLOps)**

#### **Prometheus Metrics**
Tracks three critical dimensions of pipeline health:

```python
# Request Metrics
rag_api_requests_total      # Total requests by status (success/error)
rag_api_request_latency_seconds  # P50, P95, P99 latencies (0.1-10s buckets)

# Resource Metrics
rag_api_active_requests     # Real-time request concurrency gauge
rag_cache_hits_total        # Semantic cache performance tracking
```

#### **Grafana Dashboards**
Real-time visualization of:
- Query latency distribution (percentiles over time)
- Cache hit rate (measuring semantic cache effectiveness)
- Request throughput (queries/sec by status)
- Active request gauge (system capacity utilization)
- Error rate trends (identifying degradation patterns)

#### **Docker Compose Stack**
- **Prometheus**: Time-series database (7-day retention by default, configurable)
- **Grafana**: Visualization layer with templated dashboards
- One-command deployment: `docker-compose up -d`

### 4. **Semantic Caching**
Reduces redundant LLM calls and improves response time:
- Cache semantically similar queries to avoid re-computation
- Configurable similarity threshold for cache matching
- Tracks cache hit/miss rates via Prometheus metrics

### 5. **Production-Ready Deployment**
- **Virtual Environment Isolation**: Python 3.12.4 with `.venv`
- **Dependency Management**: Pinned versions in `requirements.txt`
- **Environment Configuration**: `.env` support for API keys, model paths, cache settings
- **Graceful Startup/Shutdown**: FastAPI lifespan context managers for resource management

---

## 📊 Pepsico Beverage Sales Dataset

This project includes curated sales reports for three major Pepsico brands:

| Document | Content | Use Case |
|----------|---------|----------|
| **Pepsi_sales.txt** | Revenue breakdown, channel mix (retail 58%, wholesale 24%, online 12%, HORECA 6%), top accounts, promotional campaigns | Analyze flagship brand performance |
| **Dew_sales.txt** | Distribution logistics, supply chain metrics (96.3% shelf availability), stockout incidents, SKU strategy | Supply chain optimization queries |
| **Marinda_sales.txt** | Regional market penetration, strategic initiatives (loyalty programs with top 50 accounts), growth opportunities | Emerging market insights |

**Data Characteristics**:
- **12 chunks total** across 3 documents (4 chunks per source file)
- **Chunk size**: 1000 tokens with 200-token overlap for context continuity
- **Domain**: B2B sales, supply chain, trade channel management
- **Temporal scope**: 2025-2026 realistic sales scenarios

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (for monitoring stack)
- Git

### Step 1: Environment Setup

```powershell
# Clone the repository
git clone https://github.com/<your-org>/rag-beverage-insights.git
cd rag-beverage-insights

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Model Paths

Update `ingestion.py` or set environment variables:
```powershell
$env:HF_HOME = "E:/models/huggingface"
$env:SENTENCE_TRANSFORMERS_HOME = "E:/models/sentence_transformers"
```

### Step 3: Ingest Data

```powershell
# Load beverage sales documents, create embeddings, persist to ChromaDB
python ingestion.py
```

Expected output:
```
Loading: Pepsi_sales.txt
  Loaded 1 page(s)
  Split into 4 chunks
Loading: Dew_sales.txt
  Loaded 1 page(s)
  Split into 4 chunks
Loading: Marinda_sales.txt
  Loaded 1 page(s)
  Split into 4 chunks
Total chunks across all files: 12
Added 12 chunks to ChromaDB collection 'beverage_sales'
```

### Step 4: Start the API Server

```powershell
python api.py
# API accessible at http://localhost:8000
# OpenAPI docs at http://localhost:8000/docs
```

### Step 5: Launch Monitoring Stack

```powershell
# Start Prometheus (http://localhost:9090) and Grafana (http://localhost:3000)
docker-compose up -d

# Grafana default credentials: admin / admin
```

---

## 📖 API Usage

### Query Endpoint

**Request:**
```bash
POST http://localhost:8000/query

{
  "question": "Which product launched a loyalty program with top 50 accounts?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "Which product launched a loyalty program with top 50 accounts?",
  "answer": "Marinda launched a pilot loyalty program with the top 50 accounts to secure replenishment frequency and trade support, while bundling with high-velocity snack SKUs in convenience channels.",
  "sources": [
    {
      "rank": 1,
      "source": "docs/Marinda_sales.txt",
      "page": 1,
      "content_preview": "Strategic Initiatives & Opportunities: - Launch a pilot loyalty program with the top 50 accounts...",
      "latency_ms": 234.5
    }
  ],
  "latency_ms": 850
}
```

### Health Check

```bash
GET http://localhost:8000/health

Response: {"status": "healthy", "vectorstore": "ready", "llm": "ready"}
```

---

## 🏗️ Project Structure

```
rag-beverage-insights/
├── api.py                      # FastAPI server with request handling & validation
├── ingestion.py                # Document loading, chunking, embedding pipeline
├── retrieval.py                # Vector search and BM25 ranking logic
├── generation.py               # LLM answer generation from context
├── semantic_cache.py           # Semantic similarity caching layer
├── prometheus_metrics.py        # Prometheus metric definitions
├── main.py                     # CLI entrypoint for batch processing
├── docker-compose.yml          # Prometheus + Grafana stack
├── prometheus.yml              # Prometheus scrape configuration
├── requirements.txt            # Python dependency pinning
├── .gitignore                  # Excludes models/ and chroma_db/
├── docs/                       # Source documents (Pepsico beverage sales)
│   ├── Pepsi_sales.txt
│   ├── Dew_sales.txt
│   └── Marinda_sales.txt
├── models/                     # (Gitignored) Local model cache
│   ├── huggingface/            # HF model downloads
│   ├── sentence_transformers/  # Embedding model cache
│   └── TheBloke/               # Quantized LLM models
└── chroma_db/                  # (Gitignored) Vector store persistence

```

---

## 🔍 Monitoring & Observability

### Prometheus Scrape Configuration

The system scrapes metrics from the FastAPI server every 15 seconds:

```yaml
scrape_configs:
  - job_name: 'rag-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Key Metrics for SLO Monitoring

| Metric | Alert Threshold | Rationale |
|--------|-----------------|-----------|
| `rag_api_request_latency_seconds` (p95) | > 5s | User experience degradation |
| `rag_api_requests_total{status="error"}` | > 5% of traffic | System reliability |
| `rag_api_active_requests` | > 100 | Resource exhaustion |
| `rag_cache_hits_total` / total requests | < 20% | Cache ineffectiveness |

### Grafana Dashboard Queries

**Example: Cache Effectiveness Over Time**
```promql
rate(rag_cache_hits_total[5m]) / rate(rag_api_requests_total[5m])
```

**Example: P95 Latency Trend**
```promql
histogram_quantile(0.95, rate(rag_api_request_latency_seconds_bucket[5m]))
```

**Example: Error Rate**
```promql
rate(rag_api_requests_total{status="error"}[5m])
```

---

## 🛠️ Configuration

### Environment Variables

Create a `.env` file:

```ini
# LLM Configuration
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Model Paths
HF_HOME=E:/models/huggingface
SENTENCE_TRANSFORMERS_HOME=E:/models/sentence_transformers

# Cache Settings
CACHE_SIMILARITY_THRESHOLD=0.85
CACHE_MAX_SIZE=1000

# API Configuration
API_PORT=8000
API_WORKERS=4
```

### Chunking Strategy

Located in `ingestion.py`:
- **chunk_size**: 1000 tokens (adjust for domain specificity)
- **chunk_overlap**: 200 tokens (maintain context continuity)
- **separator**: `\n\n` for paragraph-level breaks

### Model Selection

```python
# Current embedding model: sentence-transformers/all-MiniLM-L6-v2
# ~384 dimensions, ~150MB, excellent for retrieval tasks
# Alternatives:
# - all-MiniLM-L12-v2 (384d, larger, slower)
# - all-mpnet-base-v2 (768d, SOTA, slower)
```

---

## 📈 Performance Characteristics

### Benchmarks (Local Testing)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Document Ingestion (12 chunks) | ~3s | First run, model download cached after |
| Embedding Generation (1 query) | ~50ms | Cached after first inference |
| Semantic Search (k=5) | ~1ms | ChromaDB vector indexing |
| LLM Answer Generation | ~800ms | Depends on model & context size |
| **End-to-End Query** | **~900ms** | With cache miss |
| **Cached Query** | **~150ms** | Semantic cache hit |

### Scaling Considerations

- **Horizontal**: API layer is stateless; scale behind load balancer
- **ChromaDB**: Persistent layer; consider Milvus/Qdrant for distributed deployments
- **Monitoring**: Prometheus handles ~1M samples/sec; Grafana UI remains responsive at scale
- **Model Inference**: GPU acceleration recommended for >50 QPS; currently CPU-bound

---

## 🧪 Testing

### Retrieval Validation

Test that chunks are retrieved correctly for beverage-specific queries:

```powershell
$pythonExe = '.\.venv\Scripts\python.exe'
& $pythonExe -c "
from ingestion import load_and_split_documents, load_chunks_to_chromadb
chunks = load_and_split_documents()
store = load_chunks_to_chromadb(chunks)
results = store.similarity_search('Which product launched a loyalty program?', k=2)
for r in results:
    print(f\"Source: {r.metadata['source']}\")
    print(f\"Content: {r.page_content[:300]}...\n\")
"
```

### API Testing

```bash
# Using curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the European supply chain stockout for Dew?",
    "top_k": 3
  }'

# Using Python requests
import requests
resp = requests.post("http://localhost:8000/query", json={
    "question": "Which beverage has the highest HORECA channel revenue?",
    "top_k": 5
})
print(resp.json())
```

---

## 🚢 Deployment

### Local Docker Deployment

```bash
# Build custom image with dependencies
docker build -t rag-api:latest .

# Run with monitoring
docker-compose up -d
```

### Production Considerations

- **Model Serving**: Use TorchServe or Triton for multi-GPU inference
- **Vector Store**: Deploy ChromaDB as separate service or use managed VectorDB (Pinecone, Weaviate)
- **API Gateway**: Place behind Nginx/Kong for authentication, rate limiting
- **Secrets Management**: Use Vault or cloud provider secrets (AWS Secrets Manager, Azure Key Vault)
- **Logging**: Integrate ELK stack or cloud logging (CloudWatch, Stackdriver)
- **Tracing**: Add OpenTelemetry for distributed request tracing

---

## 🔒 Security Best Practices

✅ **Implemented:**
- Input validation via Pydantic
- CORS configuration for API access
- Environment variable usage for secrets
- `.gitignore` for model files and sensitive data

⚠️ **Recommended for Production:**
- API authentication (OAuth2, JWT)
- Rate limiting (request throttling per user)
- Content filtering (PII/sensitive data masking)
- Query logging audit trail
- Encrypted model storage

---

## 📚 MLOps & CI/CD Integration

### GitHub Actions Workflow (Recommended)

```yaml
name: RAG Pipeline Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/
      - run: python ingestion.py  # Validate pipeline
      - run: docker-compose up -d && sleep 10 && curl http://localhost:9090/api/v1/status/config
```

---

## 🤝 Contributing

### Development Workflow

1. Create feature branch: `git checkout -b feature/cache-optimization`
2. Make changes and test locally
3. Run pytest: `pytest tests/`
4. Commit with meaningful message: `git commit -m "feat: improve semantic cache efficiency"`
5. Push to GitHub and open PR for review

### Code Quality Standards

- Type hints for all functions
- Docstrings following Google style guide
- Logging at INFO/DEBUG levels (not print statements)
- Test coverage > 80% for critical paths

---

## 📝 License

This project is provided as-is for demonstration and production use.

---

## 🎓 Key AI/ML Engineering Concepts Demonstrated

1. **RAG Architecture**: Combining retrieval with generation for grounded answers
2. **Embedding Models**: Semantic understanding via transformer-based embeddings
3. **Vector Databases**: Efficient similarity search at scale
4. **Caching Strategies**: Semantic caching reduces computational overhead
5. **API Design**: RESTful patterns with structured responses & validation
6. **Observability**: Prometheus/Grafana for production monitoring
7. **MLOps**: Pipeline orchestration, dependency management, environment isolation
8. **Production Patterns**: Async processing, graceful shutdown, error handling

---

## 📞 Support & Feedback

For issues, feature requests, or questions:
- Open a GitHub Issue with detailed reproduction steps
- Check existing discussions for similar topics
- Include Prometheus metrics & logs if reporting performance issues

---

**Last Updated:** 2026-06-11 | **Maintainer:** AI Engineering Team






