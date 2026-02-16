"""
Phase 3 - Script 02: Search Modalities in ChromaDB
====================================================

This script demonstrates four different approaches to searching in ChromaDB:

1. Semantic search -- find documents by meaning similarity
2. Document content filter -- filter by text content ($contains)
3. Metadata filter -- filter by structured metadata fields
4. Combined -- semantic search + metadata filtering together

We load the full dealership dataset (inventory, policies, promotions, FAQ)
and show how each approach returns different results for the same information
need. Understanding when to use each approach is critical for building
effective RAG systems.
"""

import csv
import json
from pathlib import Path

import chromadb


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


def load_inventory(path: Path = DATA_DIR / "inventory.csv") -> list[dict]:
    """Load inventory CSV and return list of document dicts."""
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (
                f"{row['year']} {row['make']} {row['model']} {row['trim']} - "
                f"${row['price']} - {row['condition']} - {row['color']} - "
                f"Features: {row['features']} - Status: {row['status']}"
            )
            metadata = {
                "source": "inventory",
                "make": row["make"],
                "model": row["model"],
                "year": int(row["year"]),
                "price": float(row["price"]),
                "category": row["category"],
                "condition": row["condition"],
                "status": row["status"],
            }
            docs.append({"text": text, "metadata": metadata})
    return docs


def load_policies(path: Path = DATA_DIR / "policies.txt") -> list[dict]:
    """Load policies text file, split by section, return document dicts."""
    content = path.read_text(encoding="utf-8")
    # Split on section headers (lines of dashes following a title)
    sections = []
    current_section = ""
    current_title = ""

    for line in content.split("\n"):
        if line.startswith("---") or line.startswith("==="):
            continue
        if line.strip() and line.strip() == line.strip().upper() and len(line.strip()) > 3:
            # This looks like a section header
            if current_section.strip():
                sections.append((current_title, current_section.strip()))
            current_title = line.strip()
            current_section = ""
        else:
            current_section += line + "\n"

    if current_section.strip():
        sections.append((current_title, current_section.strip()))

    docs = []
    for title, body in sections:
        if not body:
            continue
        docs.append({
            "text": f"{title}\n{body}",
            "metadata": {
                "source": "policies",
                "section": title.lower().replace(" ", "_"),
            },
        })
    return docs


def load_promotions(path: Path = DATA_DIR / "promotions.md") -> list[dict]:
    """Load promotions markdown, split by ## headings, return document dicts."""
    content = path.read_text(encoding="utf-8")
    sections = content.split("\n## ")
    docs = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        lines = section.split("\n")
        title = lines[0].strip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        if body:
            docs.append({
                "text": f"{title}: {body}",
                "metadata": {
                    "source": "promotions",
                    "promotion": title.lower().replace(" ", "_"),
                },
            })
    return docs


def load_faq(path: Path = DATA_DIR / "faq.json") -> list[dict]:
    """Load FAQ JSON and return document dicts."""
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    docs = []
    for entry in entries:
        text = f"Q: {entry['question']}\nA: {entry['answer']}"
        docs.append({
            "text": text,
            "metadata": {
                "source": "faq",
                "faq_category": entry["category"],
            },
        })
    return docs


def build_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Load all data sources and build a single ChromaDB collection."""
    try:
        client.delete_collection("search_modalities_demo")
    except Exception:
        pass

    collection = client.create_collection(name="search_modalities_demo")

    all_docs = []
    all_docs.extend(load_inventory())
    all_docs.extend(load_policies())
    all_docs.extend(load_promotions())
    all_docs.extend(load_faq())

    documents = [d["text"] for d in all_docs]
    metadatas = [d["metadata"] for d in all_docs]
    ids = [f"doc_{i:03d}" for i in range(len(all_docs))]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    # Print summary by source
    source_counts = {}
    for m in metadatas:
        source_counts[m["source"]] = source_counts.get(m["source"], 0) + 1
    print("Collection built successfully:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} documents")
    print(f"  Total: {collection.count()} documents\n")

    return collection


# ---------------------------------------------------------------------------
# Search demonstrations
# ---------------------------------------------------------------------------

def print_results(results: dict, max_doc_len: int = 100):
    """Print query results in a clean format."""
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for i, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas), 1):
        source = meta.get("source", "unknown")
        truncated = doc[:max_doc_len].replace("\n", " ")
        if len(doc) > max_doc_len:
            truncated += "..."
        print(f"  [{i}] (dist: {dist:.4f}) [{source}] {truncated}")
    print()


def demo_semantic_search(collection: chromadb.Collection):
    """Demo 1: Pure semantic search -- finding by meaning."""
    print("=" * 90)
    print("DEMO 1: Semantic Search")
    print("=" * 90)
    print("Approach: Query by meaning. No filters -- let the embedding model find")
    print("the most semantically similar documents.\n")

    query = "fuel efficient family car"
    print(f'Query: "{query}"')
    print(f"Method: collection.query(query_texts=[...], n_results=5)\n")
    print("Results:")

    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    print_results(results)

    print("Observation: Semantic search found the Toyota Camry (family sedan) and")
    print("similar vehicles even though the query didn't mention any specific make")
    print("or model. It understood 'fuel efficient family car' as a concept.\n")


def demo_document_filter(collection: chromadb.Collection):
    """Demo 2: Document content filtering with where_document."""
    print("=" * 90)
    print("DEMO 2: Document Content Filter (where_document)")
    print("=" * 90)
    print("Approach: Filter documents that contain specific text, then rank by")
    print("semantic similarity.\n")

    query = "best vehicle available"
    filter_term = "Toyota"
    print(f'Query: "{query}"')
    print(f'Filter: where_document={{\"$contains\": \"{filter_term}\"}}')
    print(f"Method: collection.query(query_texts=[...], where_document=..., n_results=5)\n")
    print("Results:")

    results = collection.query(
        query_texts=[query],
        where_document={"$contains": filter_term},
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    print_results(results)

    print(f"Observation: Only documents containing the word '{filter_term}' were returned.")
    print("This acts like a keyword pre-filter before semantic ranking. Useful when")
    print("you know the user is asking about a specific brand or term.\n")


def demo_metadata_filter(collection: chromadb.Collection):
    """Demo 3: Metadata filtering with where clause."""
    print("=" * 90)
    print("DEMO 3: Metadata Filter (where)")
    print("=" * 90)
    print("Approach: Filter by structured metadata fields, then rank by semantic")
    print("similarity within the filtered subset.\n")

    query = "best SUV for my family"
    print(f'Query: "{query}"')
    print(f'Filter: where={{\"make\": \"Toyota\"}}')
    print(f"Method: collection.query(query_texts=[...], where=..., n_results=5)\n")
    print("Results:")

    results = collection.query(
        query_texts=[query],
        where={"make": "Toyota"},
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    print_results(results)

    print("Observation: Only documents with metadata field make='Toyota' were searched.")
    print("Unlike where_document, this uses structured data -- it won't accidentally")
    print("match a document that says 'Unlike Toyota...' in the text. Metadata filters")
    print("give you precise, reliable filtering.\n")


def demo_combined_search(collection: chromadb.Collection):
    """Demo 4: Combined semantic search + metadata filtering."""
    print("=" * 90)
    print("DEMO 4: Combined Semantic Search + Metadata Filter")
    print("=" * 90)
    print("Approach: Apply metadata filters to narrow the search space, then use")
    print("semantic similarity to rank within that subset. This gives you the best")
    print("of both worlds: precision from filters and relevance from embeddings.\n")

    query = "affordable compact car with good features"
    print(f'Query: "{query}"')
    print(f'Filter: where={{\"$and\": [{{"source": "inventory"}}, {{"status": "Available"}}]}}')
    print(f"Method: collection.query(query_texts=[...], where=..., n_results=5)\n")
    print("Results:")

    results = collection.query(
        query_texts=[query],
        where={"$and": [{"source": "inventory"}, {"status": "Available"}]},
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    print_results(results)

    print("Observation: Results are limited to available inventory items (no policies,")
    print("no promotions, no FAQ, no sold/reserved cars), then ranked by semantic")
    print("relevance to 'affordable compact car with good features'. This is the")
    print("pattern you will use most often in production RAG systems.\n")


def demo_comparison(collection: chromadb.Collection):
    """Compare all four approaches side by side for the same information need."""
    print("=" * 90)
    print("COMPARISON: Same Question, Four Approaches")
    print("=" * 90)
    print("Information need: 'I want a Toyota SUV'\n")

    query = "I want a Toyota SUV"

    # Approach 1: Pure semantic
    print("--- Approach 1: Pure Semantic ---")
    r1 = collection.query(
        query_texts=[query], n_results=3,
        include=["documents", "distances", "metadatas"],
    )
    print_results(r1, max_doc_len=80)

    # Approach 2: Document content filter
    print("--- Approach 2: where_document contains 'Toyota' ---")
    r2 = collection.query(
        query_texts=[query],
        where_document={"$contains": "Toyota"},
        n_results=3,
        include=["documents", "distances", "metadatas"],
    )
    print_results(r2, max_doc_len=80)

    # Approach 3: Metadata filter
    print("--- Approach 3: where make='Toyota' ---")
    r3 = collection.query(
        query_texts=[query],
        where={"make": "Toyota"},
        n_results=3,
        include=["documents", "distances", "metadatas"],
    )
    print_results(r3, max_doc_len=80)

    # Approach 4: Combined
    print("--- Approach 4: where make='Toyota' AND source='inventory' ---")
    r4 = collection.query(
        query_texts=[query],
        where={"$and": [{"make": "Toyota"}, {"source": "inventory"}]},
        n_results=3,
        include=["documents", "distances", "metadatas"],
    )
    print_results(r4, max_doc_len=80)

    print("Notice how each approach refines the results. Pure semantic search may")
    print("return non-Toyota vehicles that are semantically similar to 'SUV'. Adding")
    print("metadata filters narrows results to exactly the right subset, while semantic")
    print("ranking ensures the best matches within that subset come first.\n")


def main():
    """Main function to run all search modality demonstrations."""
    print("Phase 3 - Search Modalities in ChromaDB")
    print("=" * 90)
    print()

    client = chromadb.Client()
    collection = build_collection(client)

    demo_semantic_search(collection)
    demo_document_filter(collection)
    demo_metadata_filter(collection)
    demo_combined_search(collection)
    demo_comparison(collection)

    print("=" * 90)
    print("KEY TAKEAWAYS")
    print("=" * 90)
    print("""
1. Semantic search is powerful for understanding intent and finding relevant
   documents even when the query uses different words.

2. where_document filters on the actual text content -- useful for ensuring
   a specific term appears, but can match unintended contexts.

3. where filters on structured metadata -- precise and reliable, but requires
   that your documents have well-designed metadata.

4. Combined search (semantic + metadata) is the most practical approach for
   production RAG systems. Use metadata to narrow the search space, then
   let semantic similarity rank the results.

5. The best approach depends on the query. Sometimes you need precision
   (metadata filter), sometimes recall (semantic search), and often both.
""")


if __name__ == "__main__":
    main()
