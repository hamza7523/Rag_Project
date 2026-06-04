import os
from pathlib import Path

# Must be set BEFORE any other imports
os.environ["HF_HOME"] = r"E:/models/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"E:/models/sentence_transformers"

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = r"E:/Rag_Project/chroma_db"
DOCS_DIR = "docs"


def load_and_split_documents(chunk_size=1000, chunk_overlap=200):
    all_chunks = []
    docs_path = Path(DOCS_DIR)

    supported_files = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.docx"))

    if not supported_files:
        print(f"No PDF or DOCX files found in {DOCS_DIR}")
        return []

    for file_path in supported_files:
        print(f"\nLoading: {file_path.name}")

        if file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))

        documents = loader.load()

        if not documents:
            print(f"  Skipping — no content found")
            continue

        print(f"  Loaded {len(documents)} page(s)")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
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


def load_chunks_to_chromadb(chunks, collection_name="fedadapriv"):
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