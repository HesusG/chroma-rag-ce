"""
01_hello_chroma.py - Your First Vector Database

This script demonstrates ChromaDB basics without any LLM.
You will learn how to:
  1. Create an in-memory ChromaDB client
  2. Create a collection (like a table in a regular database)
  3. Add documents with metadata
  4. Query using natural language (semantic search)

No API key required -- ChromaDB uses its own built-in embedding model.
"""

import os
import chromadb


def load_inventory(filepath: str) -> list[dict]:
    """
    Parse inventory.txt into a list of car dictionaries.

    The file has car entries separated by '---' lines. Each entry has:
      - Line 1: Car name (e.g., "2024 Toyota Camry LE")
      - Subsequent lines: Key: Value pairs (Price, Mileage, Color, Features, Status)

    Returns a list of dicts with keys: name, price, mileage, color, features, status,
    and parsed fields: make, model, year.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Split on "---" separator lines
    blocks = content.split("---")
    cars = []

    for block in blocks:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]

        if not lines:
            continue

        # Skip header lines (title and === underlines)
        # Find the first line that looks like a car name (starts with a year)
        car_name = None
        field_lines = []

        for line in lines:
            if car_name is None:
                # Look for a line starting with a 4-digit year
                if len(line) >= 4 and line[:4].isdigit():
                    car_name = line
                # Skip header/title lines
                continue
            else:
                field_lines.append(line)

        if car_name is None:
            continue

        # Parse the year, make, and model from the car name
        # Format: "2024 Toyota Camry LE"
        parts = car_name.split()
        year = parts[0] if parts else ""
        make = parts[1] if len(parts) > 1 else ""
        model = " ".join(parts[2:]) if len(parts) > 2 else ""

        # Parse key-value fields
        car = {
            "name": car_name,
            "year": year,
            "make": make,
            "model": model,
        }

        for line in field_lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                car[key] = value

        cars.append(car)

    return cars


def parse_price(price_str: str) -> int:
    """Convert a price string like '$28,990' to an integer like 28990."""
    return int(price_str.replace("$", "").replace(",", ""))


def main():
    print("=" * 60)
    print("  01_hello_chroma.py -- Your First Vector Database")
    print("=" * 60)
    print()

    # ----------------------------------------------------------------
    # Step 1: Create a ChromaDB client (in-memory)
    # ----------------------------------------------------------------
    print("[Step 1] Creating an in-memory ChromaDB client...")
    print("  This is 'ephemeral' mode -- data lives only while the script runs.")
    print("  No files are saved to disk.")
    print()

    client = chromadb.Client()  # ephemeral, in-memory

    # ----------------------------------------------------------------
    # Step 2: Create a collection
    # ----------------------------------------------------------------
    print("[Step 2] Creating a collection called 'car_inventory'...")
    print("  A collection is like a table, but it stores documents + embeddings.")
    print("  ChromaDB will automatically generate embeddings for each document")
    print("  using its built-in embedding model.")
    print()

    collection = client.create_collection(name="car_inventory")

    # ----------------------------------------------------------------
    # Step 3: Load car data from inventory.txt
    # ----------------------------------------------------------------
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "inventory.txt")
    print(f"[Step 3] Loading car inventory from: {data_path}")

    cars = load_inventory(data_path)
    print(f"  Found {len(cars)} cars in inventory.")
    print()

    # ----------------------------------------------------------------
    # Step 4: Add documents to the collection
    # ----------------------------------------------------------------
    print("[Step 4] Adding cars to the ChromaDB collection...")
    print("  Each car becomes a 'document' (text) with metadata (structured fields).")
    print("  ChromaDB converts each document into an embedding vector automatically.")
    print()

    documents = []
    metadatas = []
    ids = []

    for i, car in enumerate(cars):
        # Build a readable document string for each car
        doc_text = car["name"]
        if "price" in car:
            doc_text += f"\nPrice: {car['price']}"
        if "mileage" in car:
            doc_text += f"\nMileage: {car['mileage']}"
        if "color" in car:
            doc_text += f"\nColor: {car['color']}"
        if "features" in car:
            doc_text += f"\nFeatures: {car['features']}"
        if "status" in car:
            doc_text += f"\nStatus: {car['status']}"

        documents.append(doc_text)

        # Build metadata for filtering
        metadata = {
            "make": car.get("make", ""),
            "model": car.get("model", ""),
            "year": int(car.get("year", 0)),
        }

        # Parse price into a number for metadata filtering
        if "price" in car:
            metadata["price"] = parse_price(car["price"])

        # Include status for filtering
        if "status" in car:
            metadata["status"] = car.get("status", "")

        metadatas.append(metadata)
        ids.append(f"car_{i}")

        print(f"  Added: {car['name']}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print()
    print(f"  Total documents in collection: {collection.count()}")
    print()

    # ----------------------------------------------------------------
    # Step 5: Query -- "What SUVs do you have?"
    # ----------------------------------------------------------------
    query_1 = "What SUVs do you have?"
    print("-" * 60)
    print(f'[Step 5] Querying: "{query_1}"')
    print("  ChromaDB converts this question into an embedding, then finds")
    print("  the documents with the most similar embeddings.")
    print()

    results = collection.query(query_texts=[query_1], n_results=3)

    print(f"  Top {len(results['documents'][0])} results:\n")
    for j, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"  Result {j + 1} (distance: {dist:.4f}):")
        for line in doc.splitlines():
            print(f"    {line}")
        print()

    print("  Note: Lower distance = closer match. ChromaDB found cars whose")
    print("  descriptions are semantically closest to 'SUVs'.")
    print()

    # ----------------------------------------------------------------
    # Step 6: Query -- "Show me something under $30,000"
    # ----------------------------------------------------------------
    query_2 = "Show me something under $30,000"
    print("-" * 60)
    print(f'[Step 6] Querying: "{query_2}"')
    print("  This query relies on semantic similarity -- ChromaDB does not do")
    print("  numeric comparisons. It finds documents whose text is closest in")
    print("  meaning to the query. Results may include cars above $30k if their")
    print("  descriptions are semantically similar.")
    print()

    results = collection.query(query_texts=[query_2], n_results=3)

    print(f"  Top {len(results['documents'][0])} results:\n")
    for j, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"  Result {j + 1} (distance: {dist:.4f}):")
        for line in doc.splitlines():
            print(f"    {line}")
        print()

    # ----------------------------------------------------------------
    # Bonus: Metadata filtering example
    # ----------------------------------------------------------------
    print("-" * 60)
    print("[Bonus] Metadata filtering: cars priced under $35,000")
    print("  Unlike semantic search, metadata filters do exact numeric comparisons.")
    print()

    results = collection.query(
        query_texts=["affordable car"],
        n_results=5,
        where={"price": {"$lt": 35000}},
    )

    print(f"  Found {len(results['documents'][0])} cars under $35,000:\n")
    for j, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"  Result {j + 1} -- {meta['make']} {meta['model']} (${meta['price']:,}):")
        for line in doc.splitlines():
            print(f"    {line}")
        print()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()
    print("  What just happened:")
    print("  1. We created an in-memory vector database (ChromaDB)")
    print("  2. We loaded 10 car descriptions and stored them as documents")
    print("  3. ChromaDB automatically generated embeddings for each document")
    print("  4. We queried using natural language -- no SQL, no keyword matching")
    print("  5. ChromaDB returned the most semantically similar documents")
    print()
    print("  Key takeaway: Semantic search finds results by MEANING, not keywords.")
    print("  This is the foundation of RAG -- retrieving relevant context for an LLM.")
    print()
    print("  Next: Run 02_basic_rag.py to add an LLM and build a full RAG pipeline.")
    print()


if __name__ == "__main__":
    main()
