"""
Phase 4 - Script 2: Separate ChromaDB Collections
===================================================

This script demonstrates how to organize documents into separate ChromaDB
collections based on access level. Instead of dumping everything into one
collection, we create:

  - external_knowledge: customer-facing documents (inventory, policies, promotions, FAQ)
  - internal_knowledge: employee-only documents (complaints, training manual)

This separation is important because:
  - Customers should not see internal complaint data
  - Searching a smaller, focused collection returns more relevant results
  - You can update internal and external knowledge independently

What you will learn:
- How to create multiple ChromaDB collections
- How to load different file formats and route them to the right collection
- How to query specific collections based on user role
- The difference between customer and employee search results

Requirements:
- chromadb (installed via requirements.txt)
- Data files in the data/ directory
"""

import os
import csv
import json
from pathlib import Path

import chromadb


# ---------------------------------------------------------------------------
# Document Loading Helpers
# ---------------------------------------------------------------------------
# These functions load different file formats and return a list of
# (content, metadata) tuples. Each tuple represents one document chunk
# that will be added to ChromaDB.

def load_text_file(filepath):
    """Load a plain text file. Returns the entire file as one document."""
    with open(filepath, "r") as f:
        content = f.read()
    return [(content, {"source": filepath.name, "type": "text"})]


def load_csv_file(filepath):
    """Load a CSV file. Each row becomes a separate document."""
    docs = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert each row to a readable string
            content = ", ".join(f"{k}: {v}" for k, v in row.items())
            metadata = {"source": filepath.name, "type": "csv"}
            # Preserve useful metadata fields for filtering
            for key in ["make", "model", "year", "price", "category", "status"]:
                if key in row:
                    metadata[key] = row[key]
            docs.append((content, metadata))
    return docs


def load_json_file(filepath):
    """Load a JSON FAQ file. Each Q&A pair becomes a separate document."""
    with open(filepath, "r") as f:
        data = json.load(f)
    docs = []
    for item in data:
        content = f"Question: {item['question']}\nAnswer: {item['answer']}"
        metadata = {"source": filepath.name, "type": "json"}
        if "category" in item:
            metadata["category"] = item["category"]
        docs.append((content, metadata))
    return docs


def load_markdown_file(filepath):
    """Load a Markdown file. Returns the entire file as one document."""
    with open(filepath, "r") as f:
        content = f.read()
    return [(content, {"source": filepath.name, "type": "markdown"})]


def load_pdf_file(filepath):
    """Load a PDF file using PyPDFLoader. Returns one document per page."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(filepath))
        pages = loader.load()
        docs = []
        for page in pages:
            docs.append((
                page.page_content,
                {
                    "source": filepath.name,
                    "type": "pdf",
                    "page": str(page.metadata.get("page", 0)),
                }
            ))
        return docs
    except Exception as e:
        print(f"  Could not load PDF ({e}). Skipping.")
        return []


def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """
    Split large documents into smaller chunks.

    Documents shorter than chunk_size are left as-is. Longer documents
    are split using LangChain's RecursiveCharacterTextSplitter, which
    tries to split on natural boundaries (paragraphs, sentences, words).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked = []
    for content, metadata in docs:
        if len(content) > chunk_size:
            chunks = splitter.split_text(content)
            for chunk in chunks:
                chunked.append((chunk, metadata.copy()))
        else:
            chunked.append((content, metadata))
    return chunked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path(__file__).parent / "data"

    print("=" * 60)
    print("Phase 4 - Separate ChromaDB Collections")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Load external documents (customer-facing)
    # -----------------------------------------------------------------------
    print("\n--- Loading External Documents ---")
    ext_docs = []

    # Inventory (CSV)
    inventory_path = data_dir / "inventory.csv"
    if inventory_path.exists():
        items = load_csv_file(inventory_path)
        ext_docs.extend(items)
        print(f"  inventory.csv: {len(items)} rows")

    # Policies (text)
    policies_path = data_dir / "policies.txt"
    if policies_path.exists():
        items = load_text_file(policies_path)
        ext_docs.extend(items)
        print(f"  policies.txt: {len(items)} document(s)")

    # Promotions (Markdown)
    promotions_path = data_dir / "promotions.md"
    if promotions_path.exists():
        items = load_markdown_file(promotions_path)
        ext_docs.extend(items)
        print(f"  promotions.md: {len(items)} document(s)")

    # FAQ (JSON)
    faq_path = data_dir / "faq.json"
    if faq_path.exists():
        items = load_json_file(faq_path)
        ext_docs.extend(items)
        print(f"  faq.json: {len(items)} Q&A pairs")

    # Chunk external documents
    ext_chunked = chunk_documents(ext_docs)
    print(f"\n  Total external documents: {len(ext_docs)}")
    print(f"  After chunking: {len(ext_chunked)} chunks")

    # -----------------------------------------------------------------------
    # Step 2: Load internal documents (employee-only)
    # -----------------------------------------------------------------------
    print("\n--- Loading Internal Documents ---")
    int_docs = []

    # Complaints (CSV)
    complaints_path = data_dir / "complaints.csv"
    if complaints_path.exists():
        items = load_csv_file(complaints_path)
        int_docs.extend(items)
        print(f"  complaints.csv: {len(items)} complaints")
    else:
        print("  complaints.csv: not found (skipping)")

    # Training manual (PDF)
    training_path = data_dir / "training_manual.pdf"
    if training_path.exists():
        items = load_pdf_file(training_path)
        int_docs.extend(items)
        print(f"  training_manual.pdf: {len(items)} pages")
    else:
        print("  training_manual.pdf: not found (skipping)")

    # Chunk internal documents
    int_chunked = chunk_documents(int_docs)
    print(f"\n  Total internal documents: {len(int_docs)}")
    print(f"  After chunking: {len(int_chunked)} chunks")

    # -----------------------------------------------------------------------
    # Step 3: Create ChromaDB collections
    # -----------------------------------------------------------------------
    print("\n--- Creating ChromaDB Collections ---")
    client = chromadb.Client()

    # External collection
    external = client.get_or_create_collection("external_knowledge")
    if ext_chunked:
        external.add(
            documents=[d[0] for d in ext_chunked],
            metadatas=[d[1] for d in ext_chunked],
            ids=[f"ext_{i}" for i in range(len(ext_chunked))],
        )
    print(f"  external_knowledge: {external.count()} documents")

    # Internal collection
    internal = client.get_or_create_collection("internal_knowledge")
    if int_chunked:
        internal.add(
            documents=[d[0] for d in int_chunked],
            metadatas=[d[1] for d in int_chunked],
            ids=[f"int_{i}" for i in range(len(int_chunked))],
        )
    print(f"  internal_knowledge: {internal.count()} documents")

    # -----------------------------------------------------------------------
    # Step 4: Demo queries
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Demo Queries")
    print("=" * 60)

    # Customer query -> external collection only
    customer_query = "What SUVs are available?"
    print(f"\n[Customer] Query: \"{customer_query}\"")
    print("-" * 40)
    results = external.query(query_texts=[customer_query], n_results=3)
    if results and results["documents"]:
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"\n  Result {i + 1} (source: {meta.get('source', 'unknown')}):")
            print(f"  {doc[:200]}{'...' if len(doc) > 200 else ''}")

    # Employee query -> internal collection
    employee_query = "What complaints have we received about service delays?"
    print(f"\n\n[Employee] Query: \"{employee_query}\"")
    print("-" * 40)
    results = internal.query(query_texts=[employee_query], n_results=3)
    if results and results["documents"]:
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"\n  Result {i + 1} (source: {meta.get('source', 'unknown')}):")
            print(f"  {doc[:200]}{'...' if len(doc) > 200 else ''}")

    # -----------------------------------------------------------------------
    # Step 5: Interactive mode selector
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("\nTry querying the collections yourself.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Select mode
    while True:
        mode = input("Select mode [customer/employee] (default: customer): ").strip().lower()
        if mode in ("", "customer"):
            mode = "customer"
            break
        elif mode == "employee":
            break
        elif mode in ("quit", "exit"):
            print("Goodbye!")
            return
        else:
            print("Please enter 'customer' or 'employee'.")

    print(f"\nMode: {mode}")
    if mode == "customer":
        print("Searching: external_knowledge only")
    else:
        print("Searching: internal_knowledge + external_knowledge")

    while True:
        query = input("\nYour question: ").strip()
        if not query or query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Search the appropriate collection(s)
        if mode == "customer":
            results = external.query(query_texts=[query], n_results=3)
            print(f"\n  [external_knowledge] Results:")
            if results and results["documents"]:
                for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                    print(f"    {i + 1}. [{meta.get('source', '?')}] {doc[:150]}...")
        else:
            # Employee: search both collections
            int_results = internal.query(query_texts=[query], n_results=3)
            ext_results = external.query(query_texts=[query], n_results=2)

            print(f"\n  [internal_knowledge] Results:")
            if int_results and int_results["documents"]:
                for i, (doc, meta) in enumerate(zip(int_results["documents"][0], int_results["metadatas"][0])):
                    print(f"    {i + 1}. [{meta.get('source', '?')}] {doc[:150]}...")

            print(f"\n  [external_knowledge] Results:")
            if ext_results and ext_results["documents"]:
                for i, (doc, meta) in enumerate(zip(ext_results["documents"][0], ext_results["metadatas"][0])):
                    print(f"    {i + 1}. [{meta.get('source', '?')}] {doc[:150]}...")


if __name__ == "__main__":
    main()
