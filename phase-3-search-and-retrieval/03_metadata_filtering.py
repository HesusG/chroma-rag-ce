"""
Phase 3 - Script 03: Deep Dive into Metadata Filtering
========================================================

This script explores ChromaDB's metadata filtering capabilities in depth,
using both customer-facing (external) and employee-only (internal) data.

Key concepts:
- Numeric range filters ($gte, $lte, $gt, $lt)
- Equality and set membership ($eq, $ne, $in, $nin)
- Logical combinators ($and, $or)
- Access-level filtering for data boundaries (external vs internal)
- Combining semantic queries with metadata filters

This is where retrieval starts to feel like a real production system:
structured filters ensure precision while embeddings handle relevance.
"""

import csv
import json
from pathlib import Path

import chromadb


DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_inventory() -> tuple[list[str], list[dict], list[str]]:
    """Load inventory CSV. Returns (documents, metadatas, ids)."""
    documents, metadatas, ids = [], [], []
    with open(DATA_DIR / "inventory.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (
                f"{row['year']} {row['make']} {row['model']} {row['trim']} - "
                f"${row['price']} - {row['condition']} - {row['color']} - "
                f"Category: {row['category']} - "
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
                "access_level": "external",
            }
            documents.append(text)
            metadatas.append(metadata)
            ids.append(f"inv_{i:03d}")
    return documents, metadatas, ids


def load_complaints() -> tuple[list[str], list[dict], list[str]]:
    """Load complaints CSV. Returns (documents, metadatas, ids)."""
    documents, metadatas, ids = [], [], []
    with open(DATA_DIR / "complaints.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (
                f"Complaint {row['id']} ({row['date']}): "
                f"{row['customer_name']} - {row['vehicle']} - "
                f"Category: {row['category']} - "
                f"{row['description']} - "
                f"Status: {row['status']} - Resolution: {row['resolution']}"
            )
            metadata = {
                "source": "complaints",
                "complaint_id": row["id"],
                "category": row["category"],
                "status": row["status"],
                "vehicle": row["vehicle"],
                "access_level": "internal",
            }
            documents.append(text)
            metadatas.append(metadata)
            ids.append(f"cmp_{i:03d}")
    return documents, metadatas, ids


def load_policies() -> tuple[list[str], list[dict], list[str]]:
    """Load policies as sections. Returns (documents, metadatas, ids)."""
    content = (DATA_DIR / "policies.txt").read_text(encoding="utf-8")
    documents, metadatas, ids = [], [], []
    sections = []
    current_title = ""
    current_body = ""

    for line in content.split("\n"):
        if line.startswith("---") or line.startswith("==="):
            continue
        if line.strip() and line.strip() == line.strip().upper() and len(line.strip()) > 3:
            if current_body.strip():
                sections.append((current_title, current_body.strip()))
            current_title = line.strip()
            current_body = ""
        else:
            current_body += line + "\n"
    if current_body.strip():
        sections.append((current_title, current_body.strip()))

    for i, (title, body) in enumerate(sections):
        if not body:
            continue
        documents.append(f"{title}\n{body}")
        metadatas.append({
            "source": "policies",
            "section": title.lower().replace(" ", "_"),
            "access_level": "external",
        })
        ids.append(f"pol_{i:03d}")

    return documents, metadatas, ids


def load_promotions() -> tuple[list[str], list[dict], list[str]]:
    """Load promotions markdown. Returns (documents, metadatas, ids)."""
    content = (DATA_DIR / "promotions.md").read_text(encoding="utf-8")
    documents, metadatas, ids = [], [], []
    sections = content.split("\n## ")

    idx = 0
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        lines = section.split("\n")
        title = lines[0].strip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        if body:
            documents.append(f"{title}: {body}")
            metadatas.append({
                "source": "promotions",
                "promotion": title.lower().replace(" ", "_"),
                "access_level": "external",
            })
            ids.append(f"prm_{idx:03d}")
            idx += 1

    return documents, metadatas, ids


def load_faq() -> tuple[list[str], list[dict], list[str]]:
    """Load FAQ JSON. Returns (documents, metadatas, ids)."""
    with open(DATA_DIR / "faq.json", encoding="utf-8") as f:
        entries = json.load(f)

    documents, metadatas, ids = [], [], []
    for i, entry in enumerate(entries):
        documents.append(f"Q: {entry['question']}\nA: {entry['answer']}")
        metadatas.append({
            "source": "faq",
            "faq_category": entry["category"],
            "access_level": "external",
        })
        ids.append(f"faq_{i:03d}")

    return documents, metadatas, ids


def build_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Build a collection with all data sources including access levels."""
    try:
        client.delete_collection("metadata_filtering_demo")
    except Exception:
        pass

    collection = client.create_collection(name="metadata_filtering_demo")

    all_docs, all_metas, all_ids = [], [], []

    for loader_name, loader_fn in [
        ("inventory", load_inventory),
        ("complaints", load_complaints),
        ("policies", load_policies),
        ("promotions", load_promotions),
        ("faq", load_faq),
    ]:
        docs, metas, ids = loader_fn()
        all_docs.extend(docs)
        all_metas.extend(metas)
        all_ids.extend(ids)
        print(f"  Loaded {len(docs):>3} documents from {loader_name}")

    collection.add(documents=all_docs, metadatas=all_metas, ids=all_ids)
    print(f"\n  Total documents in collection: {collection.count()}")

    # Count by access level
    external = sum(1 for m in all_metas if m.get("access_level") == "external")
    internal = sum(1 for m in all_metas if m.get("access_level") == "internal")
    print(f"  External (customer-facing): {external}")
    print(f"  Internal (employee-only):   {internal}")
    print()

    return collection


# ---------------------------------------------------------------------------
# Filter demonstrations
# ---------------------------------------------------------------------------

def print_results(results: dict, max_doc_len: int = 120):
    """Print query results in a clean format."""
    if not results["documents"][0]:
        print("  (no results)\n")
        return

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for i, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas), 1):
        source = meta.get("source", "?")
        access = meta.get("access_level", "?")
        truncated = doc[:max_doc_len].replace("\n", " ")
        if len(doc) > max_doc_len:
            truncated += "..."
        print(f"  [{i}] (dist: {dist:.4f}) [{source}|{access}] {truncated}")
    print()


def demo_filter_by_year(collection: chromadb.Collection):
    """Filter inventory by model year."""
    print("=" * 90)
    print("FILTER 1: Vehicles from 2024 or newer")
    print("=" * 90)
    print('Filter: where={"year": {"$gte": 2024}}')
    print('Query:  "new car"\n')

    results = collection.query(
        query_texts=["new car"],
        where={"year": {"$gte": 2024}},
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    print("Results:")
    print_results(results)

    count = len(results["documents"][0])
    print(f"  Returned {count} results, all with year >= 2024.")
    print("  The year filter is applied BEFORE semantic ranking.\n")


def demo_filter_by_price_range(collection: chromadb.Collection):
    """Filter inventory by price range using $and."""
    print("=" * 90)
    print("FILTER 2: Vehicles priced between $20,000 and $35,000")
    print("=" * 90)
    print('Filter: where={"$and": [{"price": {"$gte": 20000}}, {"price": {"$lte": 35000}}]}')
    print('Query:  "affordable car"\n')

    results = collection.query(
        query_texts=["affordable car"],
        where={"$and": [{"price": {"$gte": 20000}}, {"price": {"$lte": 35000}}]},
        n_results=10,
        include=["documents", "distances", "metadatas"],
    )
    print("Results:")
    print_results(results)

    # Show prices for verification
    if results["metadatas"][0]:
        prices = [m.get("price", "N/A") for m in results["metadatas"][0] if "price" in m]
        if prices:
            print(f"  Price verification: {['${:,.0f}'.format(p) for p in prices]}")
            print(f"  All prices are in the $20,000-$35,000 range.\n")


def demo_filter_by_category(collection: chromadb.Collection):
    """Filter inventory by vehicle category."""
    print("=" * 90)
    print("FILTER 3: SUVs only")
    print("=" * 90)
    print('Filter: where={"category": "suv"}')
    print('Query:  "family vehicle with space"\n')

    results = collection.query(
        query_texts=["family vehicle with space"],
        where={"category": "suv"},
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    print("Results:")
    print_results(results)

    if results["metadatas"][0]:
        categories = [m.get("category", "?") for m in results["metadatas"][0]]
        print(f"  Category verification: {categories}")
        print(f"  All results are SUVs, ranked by semantic relevance to 'family vehicle'.\n")


def demo_filter_by_access_level(collection: chromadb.Collection):
    """Show how access-level filtering controls data visibility."""
    print("=" * 90)
    print("FILTER 4: Access Level -- External vs. Internal")
    print("=" * 90)
    print("This is the most important filter for production RAG systems.")
    print("It controls what data different user types can see.\n")

    query = "customer complaints about vehicle issues"

    # External (customer-facing) view
    print(f'Query: "{query}"')
    print('Filter: where={"access_level": "external"} (customer view)\n')

    results_external = collection.query(
        query_texts=[query],
        where={"access_level": "external"},
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    print("Results (EXTERNAL -- what a customer sees):")
    print_results(results_external)

    # Internal (employee) view
    print(f'Same query: "{query}"')
    print('Filter: where={"access_level": "internal"} (employee view)\n')

    results_internal = collection.query(
        query_texts=[query],
        where={"access_level": "internal"},
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    print("Results (INTERNAL -- what an employee sees):")
    print_results(results_internal)

    print("  Key insight: The same query returns completely different results based on")
    print("  the access_level filter. Customers see FAQ and policy documents about")
    print("  complaint processes. Employees see actual complaint records with names,")
    print("  details, and resolutions.")
    print()
    print("  This pattern enforces data boundaries at the retrieval layer, preventing")
    print("  internal information from leaking into customer-facing responses.\n")


def demo_complaint_filters(collection: chromadb.Collection):
    """Filter complaints by status (internal data)."""
    print("=" * 90)
    print("FILTER 5: Open complaints (employee view)")
    print("=" * 90)
    print('Filter: where={"$and": [{"access_level": "internal"}, {"status": "open"}]}')
    print('Query:  "unresolved vehicle problems"\n')

    results = collection.query(
        query_texts=["unresolved vehicle problems"],
        where={"$and": [{"access_level": "internal"}, {"status": "open"}]},
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    print("Results:")
    print_results(results)

    if results["metadatas"][0]:
        statuses = [m.get("status", "?") for m in results["metadatas"][0]]
        print(f"  Status verification: {statuses}")
        print("  Only open complaints returned -- useful for a service dashboard.\n")


def demo_combined_semantic_and_filter(collection: chromadb.Collection):
    """Combine a specific semantic query with multiple metadata filters."""
    print("=" * 90)
    print("FILTER 6: Combined -- Semantic Query + Multiple Filters")
    print("=" * 90)
    print("Scenario: Customer wants an affordable, available, new SUV.")
    print('Filter: where={"$and": [')
    print('    {"category": "suv"},')
    print('    {"status": "Available"},')
    print('    {"condition": "New"},')
    print('    {"price": {"$lte": 36000}}')
    print("  ]}")
    print('Query:  "affordable SUV with modern safety features"\n')

    results = collection.query(
        query_texts=["affordable SUV with modern safety features"],
        where={
            "$and": [
                {"category": "suv"},
                {"status": "Available"},
                {"condition": "New"},
                {"price": {"$lte": 36000}},
            ]
        },
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    print("Results:")
    print_results(results)

    if results["metadatas"][0]:
        for meta in results["metadatas"][0]:
            if "make" in meta:
                print(f"  - {meta.get('make')} {meta.get('model')}: "
                      f"${meta.get('price', 0):,.0f}, {meta.get('condition')}, "
                      f"{meta.get('status')}, {meta.get('category')}")
        print()
        print("  All results satisfy every filter condition AND are ranked by semantic")
        print("  relevance to 'affordable SUV with modern safety features'.\n")


def main():
    """Main function to run all metadata filtering demonstrations."""
    print("Phase 3 - Deep Dive into Metadata Filtering")
    print("=" * 90)
    print()

    print("Building collection with external and internal data...\n")
    client = chromadb.Client()
    collection = build_collection(client)

    demo_filter_by_year(collection)
    demo_filter_by_price_range(collection)
    demo_filter_by_category(collection)
    demo_filter_by_access_level(collection)
    demo_complaint_filters(collection)
    demo_combined_semantic_and_filter(collection)

    # Summary
    print("=" * 90)
    print("METADATA FILTERING CHEAT SHEET")
    print("=" * 90)
    print("""
  Operator        | Example                                   | Use Case
  ----------------|-------------------------------------------|---------------------------
  $eq (default)   | {"make": "Toyota"}                        | Exact match
  $ne             | {"status": {"$ne": "Sold"}}               | Exclude a value
  $gt / $gte      | {"year": {"$gte": 2024}}                  | Numeric lower bound
  $lt / $lte      | {"price": {"$lte": 35000}}                | Numeric upper bound
  $in             | {"make": {"$in": ["Toyota", "Honda"]}}     | Match any in list
  $nin            | {"category": {"$nin": ["truck"]}}          | Exclude list values
  $and            | {"$and": [{...}, {...}]}                   | All conditions must match
  $or             | {"$or": [{...}, {...}]}                    | Any condition can match
  $contains       | where_document={"$contains": "leather"}   | Text must contain term
  $not_contains   | where_document={"$not_contains": "sold"}  | Text must not contain term

Tips:
  - Always add access_level metadata to enforce data boundaries
  - Use numeric types (int, float) for year and price to enable range queries
  - Combine metadata filters with semantic queries for the best results
  - Test your filters in isolation before combining them
""")


if __name__ == "__main__":
    main()
