"""
02_basic_rag.py - Your First RAG Pipeline

This script builds a complete Retrieval-Augmented Generation pipeline:
  1. Load documents (inventory + policies) into ChromaDB
  2. Accept a user question
  3. Retrieve relevant documents from ChromaDB
  4. Send the question + retrieved context to an LLM (via Together.ai)
  5. Print the LLM's grounded answer

This is the core RAG loop that powers most AI chatbots with custom data.

Requirements:
  - TOGETHER_API_KEY set in .env file or as an environment variable
  - data/inventory.txt and data/policies.txt in the same directory

Usage:
  python 02_basic_rag.py
"""

import os
import sys

from dotenv import load_dotenv
import chromadb
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, SystemMessage


def load_text_blocks(filepath: str) -> list[str]:
    """
    Load a text file and split it into blocks separated by '---' lines.

    Each block becomes a separate document in the vector database.
    This is a simple chunking strategy -- each logical section (a car listing
    or a policy) becomes its own retrievable unit.
    """
    with open(filepath, "r") as f:
        content = f.read()

    blocks = content.split("---")
    documents = []

    for block in blocks:
        text = block.strip()
        if text and not all(c in "=- \n" for c in text):
            # Skip blocks that are only header decorations
            # Check if the block has substantive content
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            # Filter out blocks that are just title/underline headers
            content_lines = [
                line for line in lines
                if not all(c in "=-" for c in line)
            ]
            if content_lines:
                documents.append(text)

    return documents


def build_collection(client: chromadb.Client) -> chromadb.Collection:
    """
    Create a ChromaDB collection and populate it with documents from
    both inventory.txt and policies.txt.

    Returns the populated collection.
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    collection = client.create_collection(name="dealership_docs")

    documents = []
    metadatas = []
    ids = []
    doc_id = 0

    # Load inventory documents
    inventory_path = os.path.join(data_dir, "inventory.txt")
    print(f"  Loading inventory from: {inventory_path}")
    inventory_blocks = load_text_blocks(inventory_path)
    for block in inventory_blocks:
        documents.append(block)
        metadatas.append({"source": "inventory", "type": "car_listing"})
        ids.append(f"doc_{doc_id}")
        doc_id += 1
    print(f"  Loaded {len(inventory_blocks)} inventory documents.")

    # Load policy documents
    policies_path = os.path.join(data_dir, "policies.txt")
    print(f"  Loading policies from: {policies_path}")
    policy_blocks = load_text_blocks(policies_path)
    for block in policy_blocks:
        documents.append(block)
        metadatas.append({"source": "policies", "type": "policy"})
        ids.append(f"doc_{doc_id}")
        doc_id += 1
    print(f"  Loaded {len(policy_blocks)} policy documents.")

    # Add all documents to ChromaDB
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"  Total documents in collection: {collection.count()}")

    return collection


def retrieve_context(collection: chromadb.Collection, query: str, n_results: int = 4) -> list[str]:
    """
    Query ChromaDB and return the most relevant documents.

    Args:
        collection: The ChromaDB collection to search.
        query: The user's question.
        n_results: How many documents to retrieve (default: 4).

    Returns:
        A list of document strings, ranked by relevance.
    """
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


def format_context(documents: list[str]) -> str:
    """
    Format retrieved documents into a single context string for the LLM.

    This is a key part of context engineering -- structuring retrieved
    information so the LLM can easily read and reference it.
    """
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"[Document {i}]\n{doc}")
    return "\n\n".join(context_parts)


def ask_rag(
    llm: ChatTogether,
    collection: chromadb.Collection,
    question: str,
    show_context: bool = True,
) -> str:
    """
    The core RAG function:
      1. Retrieve relevant documents from ChromaDB
      2. Format them into context
      3. Send question + context to the LLM
      4. Return the answer

    Args:
        llm: The ChatTogether LLM instance.
        collection: The ChromaDB collection to search.
        question: The user's question.
        show_context: Whether to print retrieved documents (for learning).

    Returns:
        The LLM's answer as a string.
    """
    # Step 1: Retrieve
    retrieved_docs = retrieve_context(collection, question)

    if show_context:
        print("\n--- Retrieved Documents ---")
        for i, doc in enumerate(retrieved_docs, 1):
            # Show a preview of each document (first 120 chars)
            preview = doc[:120].replace("\n", " ")
            if len(doc) > 120:
                preview += "..."
            print(f"  [{i}] {preview}")
        print("--- End of Retrieved Documents ---\n")

    # Step 2: Format context
    context = format_context(retrieved_docs)

    # Step 3: Build the prompt
    system_prompt = (
        "You are a helpful sales assistant for AutoElite Motors, a car dealership. "
        "Answer the customer's question using ONLY the information provided in the "
        "context below. If the context does not contain enough information to answer "
        "the question, say so honestly -- do not make up information.\n\n"
        "Be friendly, concise, and helpful. If the customer asks about a specific "
        "vehicle, include relevant details like price, features, and availability.\n\n"
        f"Context:\n{context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    # Step 4: Generate answer
    response = llm.invoke(messages)
    return response.content


def main():
    print("=" * 60)
    print("  02_basic_rag.py -- Your First RAG Pipeline")
    print("=" * 60)
    print()

    # ----------------------------------------------------------------
    # Step 1: Load environment variables
    # ----------------------------------------------------------------
    print("[Step 1] Loading environment variables...")
    load_dotenv()

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print()
        print("  ERROR: TOGETHER_API_KEY not found.")
        print("  Set it in a .env file or export it in your terminal:")
        print()
        print('    export TOGETHER_API_KEY="your-key-here"')
        print()
        print("  Get a free key at https://api.together.ai/")
        sys.exit(1)

    print("  API key loaded successfully.")
    print()

    # ----------------------------------------------------------------
    # Step 2: Set up ChromaDB and load documents
    # ----------------------------------------------------------------
    print("[Step 2] Setting up ChromaDB and loading documents...")
    client = chromadb.Client()
    collection = build_collection(client)
    print()

    # ----------------------------------------------------------------
    # Step 3: Initialize the LLM
    # ----------------------------------------------------------------
    model_name = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    print(f"[Step 3] Initializing LLM: {model_name}")
    print("  Using Together.ai as the LLM provider.")
    print()

    llm = ChatTogether(
        model=model_name,
        temperature=0.3,
        max_tokens=1024,
    )

    # ----------------------------------------------------------------
    # Step 4: Interactive chat loop
    # ----------------------------------------------------------------
    print("=" * 60)
    print("  AutoElite Motors -- RAG-Powered Assistant")
    print("=" * 60)
    print()
    print("  Ask questions about our car inventory and dealership policies.")
    print("  The assistant retrieves relevant documents, then generates an answer.")
    print()
    print("  Example questions:")
    print('    - "What SUVs do you have?"')
    print('    - "Tell me about your financing options"')
    print('    - "Do you have anything under $30,000?"')
    print('    - "What is your return policy?"')
    print()
    print('  Type "quit" or "exit" to stop.')
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("\nGoodbye! Thanks for visiting AutoElite Motors.")
            break

        try:
            answer = ask_rag(llm, collection, question)
            print(f"Assistant: {answer}")
            print()
        except Exception as e:
            print(f"\n  Error: {e}")
            print("  Make sure your TOGETHER_API_KEY is valid and you have internet access.")
            print()


if __name__ == "__main__":
    main()
