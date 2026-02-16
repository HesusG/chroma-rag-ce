"""
Phase 3 - Script 01: Understanding Embedding Models and Distance Scores
========================================================================

This script demonstrates how ChromaDB's default embeddings work and what
the distance scores returned by queries actually mean.

Key concepts:
- ChromaDB uses the all-MiniLM-L6-v2 sentence-transformer model by default
- The default distance metric is L2 (squared Euclidean distance)
- Lower distance = more semantically similar
- Distance of 0 = identical text
- Typical useful results have distances below ~1.5; above that relevance drops

We create a collection of car-related documents, run several test queries,
and show a formatted comparison of results with their distances.
"""

import chromadb


def create_car_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Create a ChromaDB collection with car-related documents."""

    # Delete if exists from a previous run
    try:
        client.delete_collection("car_embeddings_demo")
    except Exception:
        pass

    collection = client.create_collection(
        name="car_embeddings_demo",
        metadata={"hnsw:space": "l2"},  # L2 distance (default)
    )

    # A diverse set of car-related documents
    documents = [
        "The 2024 Toyota Camry LE is a fuel-efficient midsize sedan with great gas mileage and a comfortable ride for families.",
        "The 2023 Ford F-150 XLT is a powerful full-size pickup truck with a 2.7L EcoBoost V6 engine and impressive towing capacity.",
        "The 2024 Tesla Model 3 Long Range is an all-electric vehicle with 358 miles of range and Autopilot driver assistance.",
        "The 2024 Honda CR-V EX-L is a spacious compact SUV with leather seats, a panoramic moonroof, and advanced safety features.",
        "AutoElite Motors offers a 7-Day Money-Back Guarantee on all vehicle purchases with a full refund policy.",
        "Financing is available starting at 2.9% APR for 36 months on new vehicles for qualified buyers.",
        "Our certified pre-owned vehicles undergo a rigorous multi-point inspection and come with extended warranty coverage.",
        "The Chevrolet Bolt EV offers an affordable entry into electric vehicle ownership with a 200-mile range.",
        "The Ford Bronco Sport Big Bend is built for adventure with terrain management and rubberized flooring.",
        "Oil change and tire rotation packages start at $49.95 with a 12-month warranty on parts and labor.",
    ]

    ids = [f"doc_{i}" for i in range(len(documents))]

    collection.add(documents=documents, ids=ids)

    print(f"Created collection with {collection.count()} documents.\n")
    return collection


def run_query(collection: chromadb.Collection, query: str, n_results: int = 5):
    """Run a query and return results with distances."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances"],
    )
    return results


def print_query_results(query: str, results: dict, top_n: int = 5):
    """Print query results in a formatted table."""
    print("=" * 90)
    print(f"QUERY: \"{query}\"")
    print("=" * 90)
    print(f"{'Rank':<6}{'Distance':<12}{'Document (truncated)'}")
    print("-" * 90)

    documents = results["documents"][0][:top_n]
    distances = results["distances"][0][:top_n]

    for i, (doc, dist) in enumerate(zip(documents, distances), 1):
        # Truncate long documents for display
        truncated = doc[:70] + "..." if len(doc) > 70 else doc
        print(f"{i:<6}{dist:<12.4f}{truncated}")

    print()

    # Interpret the distances
    best_dist = distances[0]
    if best_dist < 0.5:
        interpretation = "Excellent match -- very high semantic similarity."
    elif best_dist < 1.0:
        interpretation = "Good match -- clearly relevant content."
    elif best_dist < 1.5:
        interpretation = "Moderate match -- somewhat related but not ideal."
    else:
        interpretation = "Weak match -- low relevance, may not be useful."

    print(f"  Interpretation: {interpretation}")
    print(f"  Best distance: {best_dist:.4f} | Worst in top-{top_n}: {distances[-1]:.4f}")
    print(f"  Spread: {distances[-1] - best_dist:.4f} (larger spread = clearer ranking)")
    print()


def demonstrate_distance_meaning(collection: chromadb.Collection):
    """Show what different distance ranges mean in practice."""
    print("\n" + "=" * 90)
    print("UNDERSTANDING DISTANCE SCORES")
    print("=" * 90)
    print("""
ChromaDB's default distance metric is L2 (squared Euclidean distance).

  Distance Range    | Meaning
  ------------------|-------------------------------------------------
  0.0 - 0.5        | Excellent match, near-identical meaning
  0.5 - 1.0        | Good match, clearly relevant content
  1.0 - 1.5        | Moderate match, somewhat related
  1.5 - 2.0        | Weak match, tangentially related at best
  2.0+             | Poor match, likely irrelevant

These ranges are approximate and depend on the embedding model and your
specific document corpus. Always inspect distances during development
to calibrate your relevance threshold.
""")

    # Demonstrate with a near-exact match vs a distant one
    print("Demonstration: Near-exact match vs. distant match")
    print("-" * 60)

    # Query that closely matches a document
    exact_results = run_query(collection, "Toyota Camry fuel efficient sedan for families", n_results=1)
    exact_dist = exact_results["distances"][0][0]
    print(f"  Query: \"Toyota Camry fuel efficient sedan for families\"")
    print(f"  Top result distance: {exact_dist:.4f}")
    print(f"  Top result: {exact_results['documents'][0][0][:70]}...")
    print()

    # Query that is semantically distant from all documents
    far_results = run_query(collection, "Italian restaurant downtown with outdoor seating", n_results=1)
    far_dist = far_results["distances"][0][0]
    print(f"  Query: \"Italian restaurant downtown with outdoor seating\"")
    print(f"  Top result distance: {far_dist:.4f}")
    print(f"  Top result: {far_results['documents'][0][0][:70]}...")
    print()

    print(f"  Distance ratio: {far_dist / exact_dist:.1f}x farther for irrelevant query")
    print(f"  This shows how distances separate relevant from irrelevant results.\n")


def main():
    """Main function to run all embedding demonstrations."""
    print("Phase 3 - Embedding Models and Distance Scores")
    print("=" * 90)
    print()

    # Initialize ChromaDB (in-memory for this demo)
    client = chromadb.Client()
    collection = create_car_collection(client)

    # --- Query 1: Semantic match for a concept ---
    results = run_query(collection, "fuel efficient family car")
    print_query_results("fuel efficient family car", results)

    # --- Query 2: Looking for a specific vehicle type ---
    results = run_query(collection, "pickup truck for towing and hauling")
    print_query_results("pickup truck for towing and hauling", results)

    # --- Query 3: Policy question ---
    results = run_query(collection, "can I return a car after buying it")
    print_query_results("can I return a car after buying it", results)

    # --- Query 4: Electric vehicle interest ---
    results = run_query(collection, "environmentally friendly zero emissions vehicle")
    print_query_results("environmentally friendly zero emissions vehicle", results)

    # --- Distance meaning demonstration ---
    demonstrate_distance_meaning(collection)

    # --- Summary comparison ---
    print("=" * 90)
    print("SUMMARY: How Semantic Search Finds Meaning, Not Just Words")
    print("=" * 90)
    print("""
Notice how the queries above found relevant documents even when the query
used completely different words:

  Query                                  | Top Match
  ---------------------------------------|-------------------------------------------
  "fuel efficient family car"            | Toyota Camry (gas mileage, families)
  "pickup truck for towing and hauling"  | Ford F-150 (towing capacity)
  "can I return a car after buying it"   | 7-Day Money-Back Guarantee
  "environmentally friendly zero emis.." | Tesla Model 3 / Bolt EV (electric)

This is the power of semantic search: it understands meaning and intent,
not just keyword overlap. The embedding model learned these associations
from training on large text corpora.

In a RAG system, this means customers can ask questions in their own words
and still get relevant documents retrieved from your knowledge base.
""")


if __name__ == "__main__":
    main()
