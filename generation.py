import os
os.environ["HF_HOME"] = r"E:/models/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"E:/models/sentence_transformers"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from retrieval import load_retriever, retrieve
import dotenv

dotenv.load_dotenv()

CHROMA_DIR = r"E:/Rag_Project/chroma_db"


llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    model_name=os.getenv("OPENAI_MODEL_NAME", "tinyllama-1.1b-chat-v1.0"),
    max_tokens=500,
    temperature=0.2,
    seed=42,
    verbose=False,
)
# ── Prompt Template ───────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Answer only using the provided context. If the answer is not in the context, say 'I don't know'. Do not make things up."),
    ("human",
     "Context:\n{context} Provide concise information only using the relevant information.\n\nQuestion: {question}")
])
def format_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("source", "?")
        page = chunk["metadata"].get("page", "?")
        # Truncate each chunk to 300 chars to stay within TinyLlama's 2048 token limit
        truncated_content = chunk["content"][:300]
        parts.append(f"[{i+1}] (Source: {source} | Page: {page})\n{truncated_content}")
    return "\n\n".join(parts)  # ← use parts, not a new comprehension over source_chunks


def answer(query, vectorstore, bm25_index, documents, metadatas):
    chunks = retrieve(query, vectorstore, bm25_index, documents, metadatas, top_k=5)
    
    context = format_context(chunks)
    messages = prompt.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    print("\nAnswer:")
    print(response.content)
    return response.content,chunks

if __name__ == "__main__":
    vectorstore, bm25_index, documents, metadatas = load_retriever()

    query = "What is meant by 3 signal noise?"
    
    answer_text, source_chunks = answer(query, vectorstore, bm25_index, documents, metadatas)

    print(f"Query: {query}")
    print(f"\nAnswer:\n{answer_text}")
    print(f"\n--- Sources used ---")
    for i, chunk in enumerate(source_chunks):
        print(f"[{i+1}] {chunk['metadata'].get('source', '?')} | page {chunk['metadata'].get('page', '?')}")














