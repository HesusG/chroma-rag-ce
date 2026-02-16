"""
Phase 2, Script 2: Chunking Strategies
========================================
Learn how different text splitters break a document into retrieval-friendly
chunks, and see first-hand how chunk size and overlap affect the results.

Why chunking matters for RAG:
  - Too BIG   -> chunks contain multiple topics, embeddings become vague,
                  retrieved text has a lot of irrelevant noise.
  - Too SMALL -> chunks lose surrounding context, the LLM cannot give a
                  complete answer because key details are in other chunks.
  - JUST RIGHT -> each chunk covers one coherent topic with enough context
                  for the LLM to generate an accurate, grounded answer.

No API key required -- this script runs entirely locally.
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"


def load_policies():
    """Load policies.txt as a single document for chunking experiments."""
    loader = TextLoader(str(DATA_DIR / "policies.txt"))
    docs = loader.load()
    return docs


def show_chunks(chunks, label, max_display=2):
    """Pretty-print the first few chunks with their lengths."""
    print(f"\n   --- {label} ---")
    print(f"   Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:max_display]):
        content = chunk.page_content
        print(f"\n   Chunk {i + 1} ({len(content)} chars):")
        print(f"   {'─' * 50}")
        # Indent each line for readability
        for line in content.split("\n"):
            print(f"   | {line}")
        print(f"   {'─' * 50}")
        print(f"   Metadata: {chunk.metadata}")


# ===========================================================================
# Strategy 1: RecursiveCharacterTextSplitter -- chunk_size=500, overlap=50
# ===========================================================================
def strategy_recursive_500(docs):
    """
    RecursiveCharacterTextSplitter is the RECOMMENDED DEFAULT for most use cases.

    It tries to split on natural boundaries in this priority order:
      1. "\n\n" (paragraph breaks)
      2. "\n"   (line breaks)
      3. " "    (spaces)
      4. ""     (individual characters -- last resort)

    With chunk_size=500 and overlap=50, we get chunks that are roughly
    paragraph-sized. The overlap ensures that if a sentence is split at
    a boundary, the next chunk still has the tail end for context.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 1: RecursiveCharacterTextSplitter (size=500, overlap=50)")
    print("=" * 70)
    print("   Splits on paragraph breaks first, then line breaks, then spaces.")
    print("   This preserves natural document structure as much as possible.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    show_chunks(chunks, "Recursive (500/50)")
    return chunks


# ===========================================================================
# Strategy 2: RecursiveCharacterTextSplitter -- chunk_size=200, overlap=20
# ===========================================================================
def strategy_recursive_200(docs):
    """
    Same algorithm, but with a much smaller chunk size. This produces more
    chunks that are more focused, but each chunk has less context.

    Use smaller chunks when:
      - Your documents have many short, independent facts (like an FAQ).
      - You want very precise retrieval at the cost of completeness.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 2: RecursiveCharacterTextSplitter (size=200, overlap=20)")
    print("=" * 70)
    print("   Same algorithm, smaller window. More chunks, less context each.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(docs)
    show_chunks(chunks, "Recursive (200/20)")
    return chunks


# ===========================================================================
# Strategy 3: CharacterTextSplitter -- chunk_size=500, overlap=50
# ===========================================================================
def strategy_character_500(docs):
    """
    CharacterTextSplitter uses a SINGLE separator (default: "\\n\\n").
    It splits only on double-newlines. If a resulting chunk is still larger
    than chunk_size, the splitter does NOT recurse into smaller separators.

    This means some chunks may exceed the target size if there are large
    paragraphs without double-newline breaks.

    Use CharacterTextSplitter when your text has very consistent formatting
    with reliable paragraph separators.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 3: CharacterTextSplitter (size=500, overlap=50)")
    print("=" * 70)
    print("   Splits only on '\\n\\n'. Less adaptive -- chunks may vary more in size.")

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n\n",
    )
    chunks = splitter.split_documents(docs)
    show_chunks(chunks, "Character (500/50)")
    return chunks


# ===========================================================================
# Comparison Summary
# ===========================================================================
def compare(results):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"   {'Strategy':<50} {'Chunks':>6}  {'Avg Len':>7}  {'Min':>5}  {'Max':>5}")
    print(f"   {'─' * 50} {'─' * 6}  {'─' * 7}  {'─' * 5}  {'─' * 5}")

    for label, chunks in results:
        lengths = [len(c.page_content) for c in chunks]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        print(f"   {label:<50} {len(chunks):>6}  {avg_len:>7.0f}  {min_len:>5}  {max_len:>5}")

    print()
    print("   KEY INSIGHTS:")
    print("   - Smaller chunk_size -> more chunks, each more focused but less context.")
    print("   - RecursiveCharacterTextSplitter produces more uniform chunk sizes")
    print("     because it adapts to the text structure at multiple levels.")
    print("   - CharacterTextSplitter may produce fewer, larger chunks because it")
    print("     only splits on one separator.")
    print("   - Overlap adds some redundancy but prevents losing information at")
    print("     chunk boundaries.")
    print()
    print("   RECOMMENDATION: Start with RecursiveCharacterTextSplitter at")
    print("   chunk_size=500, overlap=50 and adjust based on retrieval quality.")
    print()


# ===========================================================================
# Main
# ===========================================================================
def main():
    print()
    print("PHASE 2 -- CHUNKING STRATEGIES")
    print("Splitting policies.txt with different strategies to see how chunk")
    print("size, overlap, and algorithm affect the results.")
    print()

    docs = load_policies()
    doc_length = len(docs[0].page_content)
    print(f"   Loaded policies.txt: 1 document, {doc_length} characters total.")

    # Run all three strategies
    chunks_r500 = strategy_recursive_500(docs)
    chunks_r200 = strategy_recursive_200(docs)
    chunks_c500 = strategy_character_500(docs)

    # Compare
    compare([
        ("Recursive (size=500, overlap=50)", chunks_r500),
        ("Recursive (size=200, overlap=20)", chunks_r200),
        ("Character (size=500, overlap=50)", chunks_c500),
    ])


if __name__ == "__main__":
    main()
