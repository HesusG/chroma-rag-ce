"""
Phase 2, Script 3: Multi-Document RAG Pipeline
================================================
Build a complete RAG pipeline that loads documents from 4 different formats,
chunks them, stores them in ChromaDB, and answers questions with source
attribution using a Together.ai LLM.

This is the capstone script for Phase 2. It ties together everything you
learned in scripts 01 and 02.

Requires: TOGETHER_API_KEY environment variable (set in .env or shell).
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import CSVLoader, TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"

TOGETHER_MODEL = os.getenv(
    "TOGETHER_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
)

COLLECTION_NAME = "autoelite_phase2"


# ===========================================================================
# Step 1: Load Documents from All Formats
# ===========================================================================
def load_all_documents():
    """
    Load documents from all 4 data files. Each loader is chosen to match
    the file format and produce the most useful Document objects.
    """
    all_docs = []

    # --- policies.txt (TextLoader) ---
    print("   Loading policies.txt ...")
    loader = TextLoader(str(DATA_DIR / "policies.txt"))
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = "policy"
        doc.metadata["source_file"] = "policies.txt"
    all_docs.extend(docs)
    print(f"   -> {len(docs)} document(s)")

    # --- inventory.csv (CSVLoader) ---
    print("   Loading inventory.csv ...")
    loader = CSVLoader(str(DATA_DIR / "inventory.csv"))
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = "inventory"
        doc.metadata["source_file"] = "inventory.csv"
    all_docs.extend(docs)
    print(f"   -> {len(docs)} document(s)")

    # --- promotions.md (TextLoader -- simpler, avoids unstructured dependency) ---
    print("   Loading promotions.md ...")
    loader = TextLoader(str(DATA_DIR / "promotions.md"))
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = "promotion"
        doc.metadata["source_file"] = "promotions.md"
    all_docs.extend(docs)
    print(f"   -> {len(docs)} document(s)")

    # --- faq.json (JSONLoader) ---
    print("   Loading faq.json ...")

    def faq_metadata(record: dict, metadata: dict) -> dict:
        metadata["question"] = record.get("question", "")
        metadata["category"] = record.get("category", "")
        metadata["doc_type"] = "faq"
        metadata["source_file"] = "faq.json"
        return metadata

    # Build content that includes both the question and answer so the
    # embedding captures the full meaning of each FAQ entry.
    faq_path = DATA_DIR / "faq.json"
    with open(faq_path, "r") as f:
        faq_data = json.load(f)

    from langchain_core.documents import Document

    docs = []
    for entry in faq_data:
        content = f"Q: {entry['question']}\nA: {entry['answer']}"
        doc = Document(
            page_content=content,
            metadata={
                "source": str(faq_path),
                "question": entry.get("question", ""),
                "category": entry.get("category", ""),
                "doc_type": "faq",
                "source_file": "faq.json",
            },
        )
        docs.append(doc)

    all_docs.extend(docs)
    print(f"   -> {len(docs)} document(s)")

    print(f"\n   Total documents loaded: {len(all_docs)}")
    return all_docs


# ===========================================================================
# Step 2: Chunk Documents
# ===========================================================================
def chunk_documents(docs):
    """
    Split all documents into retrieval-friendly chunks using
    RecursiveCharacterTextSplitter. CSV rows and FAQ entries are already
    small, so they will pass through mostly unchanged. The policies and
    promotions documents will be split into smaller pieces.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"   Chunks after splitting: {len(chunks)}")
    return chunks


# ===========================================================================
# Step 3: Store in ChromaDB
# ===========================================================================
def create_collection(chunks):
    """
    Create a ChromaDB collection and add all chunks with their text,
    metadata, and auto-generated IDs. ChromaDB will compute embeddings
    automatically using its default embedding function.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if it exists (fresh start each run)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "AutoElite Motors - Phase 2 multi-doc RAG"},
    )

    # Prepare data for ChromaDB
    ids = [f"doc_{i}" for i in range(len(chunks))]
    documents = [chunk.page_content for chunk in chunks]

    # ChromaDB metadata values must be str, int, float, or bool.
    # Filter metadata to only include compatible types.
    metadatas = []
    for chunk in chunks:
        clean_meta = {}
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_meta[key] = value
        metadatas.append(clean_meta)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"   Added {collection.count()} chunks to ChromaDB collection '{COLLECTION_NAME}'")
    return collection


# ===========================================================================
# Step 4: Set Up the LLM
# ===========================================================================
def create_llm():
    """Create a ChatTogether LLM instance."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("\n   ERROR: TOGETHER_API_KEY not found.")
        print("   Set it with: export TOGETHER_API_KEY='your-key-here'")
        print("   Or add it to a .env file in this directory.")
        raise SystemExit(1)

    llm = ChatTogether(
        model=TOGETHER_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )
    print(f"   LLM ready: {TOGETHER_MODEL}")
    return llm


# ===========================================================================
# Step 5: RAG Function
# ===========================================================================
def rag_query(question, collection, llm, n_results=4):
    """
    The core RAG pipeline:
      1. Query ChromaDB with the user's question to find relevant chunks.
      2. Format the retrieved chunks as context (with source attribution).
      3. Send the context + question to the LLM.
      4. Return the answer and the retrieved sources.
    """
    # --- Retrieve ---
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # --- Format context with source info ---
    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        source = meta.get("source_file", "unknown")
        doc_type = meta.get("doc_type", "unknown")
        context_parts.append(
            f"[Source {i + 1}: {source} ({doc_type}), relevance: {1 - dist:.2f}]\n{doc}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # --- Build prompt ---
    system_prompt = (
        "You are a helpful assistant for AutoElite Motors, a car dealership. "
        "Answer the customer's question based ONLY on the provided context. "
        "If the context does not contain enough information to answer, say so. "
        "When possible, mention which source the information came from."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # --- Call LLM ---
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)

    return response.content, documents, metadatas, distances


# ===========================================================================
# Step 6: Interactive CLI Loop
# ===========================================================================
def interactive_loop(collection, llm):
    """Run an interactive question-answering loop."""
    print("\n" + "=" * 70)
    print("INTERACTIVE RAG CHAT")
    print("=" * 70)
    print("Ask questions about AutoElite Motors inventory, policies,")
    print("promotions, or FAQs. Type 'quit' or 'exit' to stop.")
    print("=" * 70)

    while True:
        print()
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        try:
            answer, docs, metas, dists = rag_query(question, collection, llm)

            # Show retrieved sources
            print(f"\n{'─' * 70}")
            print("Retrieved Sources:")
            for i, (meta, dist) in enumerate(zip(metas, dists)):
                source = meta.get("source_file", "unknown")
                doc_type = meta.get("doc_type", "unknown")
                relevance = 1 - dist
                extra = ""
                if "question" in meta and meta["question"]:
                    extra = f" | FAQ: {meta['question']}"
                print(f"  [{i + 1}] {source} ({doc_type}) - relevance: {relevance:.2f}{extra}")

            # Show answer
            print(f"\n{'─' * 70}")
            print(f"Assistant: {answer}")
            print(f"{'─' * 70}")

        except Exception as e:
            print(f"\n   Error: {e}")
            print("   Please try again.")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print()
    print("PHASE 2 -- MULTI-DOCUMENT RAG PIPELINE")
    print("Loading documents from 4 formats, chunking, storing in ChromaDB,")
    print("and answering questions with source attribution.")
    print()

    # Step 1: Load
    print("Step 1: Loading documents...")
    all_docs = load_all_documents()

    # Step 2: Chunk
    print("\nStep 2: Chunking documents...")
    chunks = chunk_documents(all_docs)

    # Step 3: Store
    print("\nStep 3: Storing in ChromaDB...")
    collection = create_collection(chunks)

    # Step 4: LLM
    print("\nStep 4: Setting up LLM...")
    llm = create_llm()

    # Step 5: Interactive loop
    interactive_loop(collection, llm)


if __name__ == "__main__":
    main()
