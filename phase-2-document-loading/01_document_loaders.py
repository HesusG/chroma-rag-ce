"""
Phase 2, Script 1: Document Loaders
====================================
Learn how LangChain loads different document formats into a unified Document
structure that can be embedded, chunked, and stored in a vector database.

Each loader produces Document objects with:
  - page_content: the text that will be embedded
  - metadata: information about where the text came from

No API key required -- this script runs entirely locally.
"""

from pathlib import Path

from langchain_community.document_loaders import CSVLoader, TextLoader, JSONLoader


# ---------------------------------------------------------------------------
# Paths -- all data files live in the data/ folder next to this script
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"


# ===========================================================================
# 1. TextLoader -- Plain Text Files
# ===========================================================================
def load_text():
    """
    TextLoader is the simplest loader. It reads the entire file as a single
    Document. The only metadata it adds automatically is the file path
    (under the key 'source').

    Best for: policies, manuals, plain-text notes -- anything without
    structured fields.
    """
    print("=" * 70)
    print("1. TextLoader -- Loading policies.txt")
    print("=" * 70)

    loader = TextLoader(str(DATA_DIR / "policies.txt"))
    docs = loader.load()

    print(f"   Documents loaded: {len(docs)}")
    print(f"   Content (first 200 chars):\n   {docs[0].page_content[:200]}")
    print(f"   Metadata: {docs[0].metadata}")
    print()
    print("   WHY: TextLoader treats the whole file as one document.")
    print("   This is fine for now -- we will chunk it later in script 02.")
    print()

    return docs


# ===========================================================================
# 2. CSVLoader -- Structured Tabular Data
# ===========================================================================
def load_csv():
    """
    CSVLoader creates one Document per row. Each row's column values are
    combined into the page_content, and the metadata includes the source
    file and the row number.

    Best for: inventory lists, product catalogs, any tabular data where
    each row is an independent record.
    """
    print("=" * 70)
    print("2. CSVLoader -- Loading inventory.csv")
    print("=" * 70)

    loader = CSVLoader(str(DATA_DIR / "inventory.csv"))
    docs = loader.load()

    print(f"   Documents loaded: {len(docs)}")
    print(f"   Content (first 200 chars):\n   {docs[0].page_content[:200]}")
    print(f"   Metadata: {docs[0].metadata}")
    print()
    print("   WHY: Each CSV row becomes its own document. This means a query")
    print("   like 'red SUV under $40k' can match against individual vehicles")
    print("   instead of the entire spreadsheet.")
    print()

    return docs


# ===========================================================================
# 3. UnstructuredMarkdownLoader (with TextLoader fallback)
# ===========================================================================
def load_markdown():
    """
    UnstructuredMarkdownLoader understands Markdown structure (headings,
    lists, etc.) and can split content by elements. However, it requires
    the 'unstructured' package which has heavy dependencies.

    If 'unstructured' is not installed, we fall back to TextLoader which
    loads the Markdown as plain text -- still perfectly usable.
    """
    print("=" * 70)
    print("3. MarkdownLoader -- Loading promotions.md")
    print("=" * 70)

    try:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader

        loader = UnstructuredMarkdownLoader(str(DATA_DIR / "promotions.md"))
        docs = loader.load()
        print("   (Using UnstructuredMarkdownLoader)")
    except (ImportError, ModuleNotFoundError):
        print("   (unstructured not installed -- falling back to TextLoader)")
        loader = TextLoader(str(DATA_DIR / "promotions.md"))
        docs = loader.load()

    print(f"   Documents loaded: {len(docs)}")
    print(f"   Content (first 200 chars):\n   {docs[0].page_content[:200]}")
    print(f"   Metadata: {docs[0].metadata}")
    print()
    print("   WHY: Markdown is common for documentation and promotional content.")
    print("   A Markdown-aware loader can preserve heading hierarchy, but plain")
    print("   TextLoader works fine when chunked with RecursiveCharacterTextSplitter.")
    print()

    return docs


# ===========================================================================
# 4. JSONLoader -- Structured JSON Data
# ===========================================================================
def load_json():
    """
    JSONLoader uses jq-style schemas to extract content from JSON files.
    This is powerful because JSON often has nested structures where only
    specific fields are useful for retrieval.

    Key parameters:
      - jq_schema: a jq expression that selects which JSON elements to process.
        '.[]' means 'iterate over every item in the top-level array'.
      - content_key: which field in each element becomes the page_content.
      - metadata_func: a function that receives the raw JSON record and returns
        a dict of metadata fields to attach to the Document.
    """
    print("=" * 70)
    print("4. JSONLoader -- Loading faq.json")
    print("=" * 70)

    def extract_metadata(record: dict, metadata: dict) -> dict:
        """Pull the question and category out of each FAQ entry as metadata."""
        metadata["question"] = record.get("question", "")
        metadata["category"] = record.get("category", "")
        return metadata

    loader = JSONLoader(
        file_path=str(DATA_DIR / "faq.json"),
        jq_schema=".[]",
        content_key="answer",
        metadata_func=extract_metadata,
    )
    docs = loader.load()

    print(f"   Documents loaded: {len(docs)}")
    print(f"   Content (first 200 chars):\n   {docs[0].page_content[:200]}")
    print(f"   Metadata: {docs[0].metadata}")
    print()
    print("   WHY: Each FAQ entry becomes a document whose page_content is the")
    print("   answer (what we want to retrieve) and whose metadata includes the")
    print("   question and category (useful for filtering and display).")
    print()

    return docs


# ===========================================================================
# Main
# ===========================================================================
def main():
    print()
    print("PHASE 2 -- DOCUMENT LOADERS")
    print("Loading the same dealership data from 4 different file formats.")
    print("Each loader produces LangChain Document objects with page_content")
    print("and metadata -- a unified interface for the rest of the pipeline.")
    print()

    text_docs = load_text()
    csv_docs = load_csv()
    md_docs = load_markdown()
    json_docs = load_json()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(text_docs) + len(csv_docs) + len(md_docs) + len(json_docs)
    print(f"   policies.txt   -> {len(text_docs):>3} document(s)  (TextLoader)")
    print(f"   inventory.csv  -> {len(csv_docs):>3} document(s)  (CSVLoader)")
    print(f"   promotions.md  -> {len(md_docs):>3} document(s)  (MarkdownLoader)")
    print(f"   faq.json       -> {len(json_docs):>3} document(s)  (JSONLoader)")
    print(f"   {'':>16}   ----")
    print(f"   Total            {total:>3} document(s)")
    print()
    print("   Next step: run 02_chunking_strategies.py to learn how to split")
    print("   these documents into smaller, retrieval-friendly chunks.")
    print()


if __name__ == "__main__":
    main()
