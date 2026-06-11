RAG_Project

Overview

This repository provides a Retrieval-Augmented Generation (RAG) pipeline for local document ingestion, embedding, and retrieval using ChromaDB and Sentence-Transformers embeddings. The architecture separates ingestion, retrieval, generation, and a small API layer so you can swap models and sources without changing the core pipeline.

Key features

- Ingest documents (PDF, DOCX, and plain text) and split into chunks with a RecursiveCharacterTextSplitter.
- Create embeddings locally using Hugging Face sentence-transformers models.
- Store embeddings in ChromaDB and perform similarity search for retrieval.
- Simple API and CLI entrypoints for generating answers from retrieved chunks.

Repository Layout

- `ingestion.py`: Loads documents from `docs/`, splits them into chunks, and builds a ChromaDB collection.
- `retrieval.py`: Retrieval helpers and search logic.
- `generation.py`: Generation code using retrieved contexts.
- `api.py` / `main.py`: Simple server wrappers around the retrieval+generation flow.
- `docs/`: Document source files. (Now contains beverage sales `.txt` datasets.)
- `models/`: Local model files (ignored from git to avoid large files).
- `chroma_db/`: Local Chroma DB persistence files.

Quickstart (development)

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure local model cache locations exist (or update env vars in `ingestion.py`). The project expects local HF cache under `E:/models/huggingface` and sentence-transformers cache under `E:/models/sentence_transformers` by default. You can change these in `ingestion.py` at the top.

3. Ingest documents and build vector store:

```powershell
python ingestion.py
```

4. Run the API / generation examples as needed:

```powershell
python api.py
# or
python main.py
```

Notes

- Large model files are intentionally kept out of git; use `models/` locally and configure model loading paths.
- If you need to track large binary models, use Git LFS (not recommended for all models) or host them separately.
- The `docs/` dataset in this repo is a set of simulated beverage sales reports used as the retrieval source.

Contributing

- Follow the existing pipeline; avoid changing chunking parameters unless you understand retrieval consequences.
- Keep heavy binaries (model files) out of git.

Contact

For questions about this repository or how to adapt the pipeline to your dataset, open an issue or contact the maintainer.






