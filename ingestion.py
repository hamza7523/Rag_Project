import os
from pathlib import Path

# Must be set BEFORE any other imports
os.environ["HF_HOME"] = r"E:/models/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"E:/models/sentence_transformers"

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.schema import Document
import re
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = r"E:/Rag_Project/chroma_db"
DOCS_DIR = "docs"


from text_utils import parse_numeric_fields as _parse_numeric_fields


def load_and_split_documents(chunk_size=1000, chunk_overlap=200):
    """Load documents from `DOCS_DIR` and produce chunks.

    For `.txt` files we treat paragraph/line blocks containing numeric
    patterns as atomic Documents and store parsed numeric fields in
    metadata to preserve numeric fidelity for downstream exact lookups.
    """
    all_chunks = []
    docs_path = Path(DOCS_DIR)

    supported_files = (
        list(docs_path.glob("*.pdf"))
        + list(docs_path.glob("*.docx"))
        + list(docs_path.glob("*.txt"))
    )

    if not supported_files:
        print(f"No supported files (PDF/DOCX/TXT) found in {DOCS_DIR}")
        return []

    for file_path in supported_files:
        print(f"\nLoading: {file_path.name}")

        if file_path.suffix.lower() == ".txt":
            # load raw text and split on paragraphs (preserve numeric lines)
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
            if not documents:
                print("  Skipping — no content found")
                continue

            print(f"  Loaded {len(documents)} page(s)")

            file_docs = []
            for doc in documents:
                # split into paragraphs (double-newline or newline)
                paragraphs = [p.strip() for p in re.split(r"\n\n+|\r\n\r\n+", doc.page_content) if p.strip()]
                for p in paragraphs:
                    numeric_fields, has_numeric = _parse_numeric_fields(p)
                    metadata = {"source": str(file_path)}
                    if has_numeric:
                        # attach parsed numeric fields so retrieval can filter exactly
                        metadata.update(numeric_fields)
                        # keep the paragraph as an atomic document (no further splitting)
                        file_docs.append(Document(page_content=p, metadata=metadata))
                    else:
                        # keep text for normal splitting
                        file_docs.append(Document(page_content=p, metadata=metadata))

            # split only the non-numeric documents using the splitter to avoid fragmenting numbers
            numeric_docs = [d for d in file_docs if d.metadata and any(k in d.metadata for k in ("currency_values","int_values","percent_values","days","units_sold"))]
            non_numeric_docs = [d for d in file_docs if d not in numeric_docs]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            split_non_numeric = text_splitter.split_documents(non_numeric_docs) if non_numeric_docs else []

            chunks = numeric_docs + split_non_numeric
            print(f"  Produced {len(chunks)} chunks (txt-aware)")

        else:
            if file_path.suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix == ".docx":
                loader = Docx2txtLoader(str(file_path))
            else:
                print(f"  Skipping unknown file type: {file_path.suffix}")
                continue

            documents = loader.load()
            if not documents:
                print(f"  Skipping — no content found")
                continue

            print(f"  Loaded {len(documents)} page(s)")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            print(f"  Split into {len(chunks)} chunks")

        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all files: {len(all_chunks)}")
    return all_chunks


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=r"E:/models/sentence_transformers"
    )


def load_chunks_to_chromadb(chunks, collection_name="beverage_sales"):
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name
    )

    print(f"Added {len(chunks)} chunks to ChromaDB collection '{collection_name}'")
    return vectorstore


def main():
    chunks = load_and_split_documents()
    if not chunks:
        return

    vectorstore = load_chunks_to_chromadb(chunks)

    print("\n=== RETRIEVAL SANITY CHECK ===")
    results = vectorstore.similarity_search(
        "What privacy mechanism is used?", k=2
    )
    for i in range(len(vectorstore._collection.get(include=["metadatas"])["metadatas"])):
        print(f"\nMetadata for chunk {i+1}:")
        print(vectorstore._collection.get(include=["metadatas"])["metadatas"][i])
    
    for i, r in enumerate(results):
        print(f"\nResult {i+1}:")
        print(r.page_content[:300])
        print(f"Source: {r.metadata.get('source', '?')} | page {r.metadata.get('page', '?')}")


if __name__ == "__main__":
    main()