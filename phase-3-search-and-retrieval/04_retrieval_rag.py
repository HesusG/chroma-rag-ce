"""
Phase 3 - Script 04: Complete RAG with Access-Controlled Retrieval
===================================================================

This script brings everything together into a complete RAG pipeline:

1. Load ALL dealership data (inventory, policies, promotions, FAQ, complaints)
2. Store in ChromaDB with access_level metadata
3. Retrieve relevant documents based on user type (customer vs employee)
4. Send retrieved context to a Together.ai LLM for answer generation
5. Interactive CLI with mode switching

This demonstrates the critical retrieval layer in RAG: the quality of
retrieved documents directly determines the quality of generated answers.

Requires:
- TOGETHER_API_KEY environment variable (or .env file)
- Optional: TOGETHER_MODEL environment variable
"""

import os
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

DATA_DIR = Path(__file__).parent / "data"

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------

def load_inventory() -> tuple[list[str], list[dict], list[str]]:
    """Load vehicle inventory as documents with metadata."""
    documents, metadatas, ids = [], [], []
    with open(DATA_DIR / "inventory.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (
                f"{row['year']} {row['make']} {row['model']} {row['trim']} - "
                f"Price: ${row['price']} - Condition: {row['condition']} - "
                f"Color: {row['color']} - Category: {row['category']} - "
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


def load_policies() -> tuple[list[str], list[dict], list[str]]:
    """Load dealership policies split by section."""
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
    """Load current promotions from markdown."""
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
    """Load FAQ entries from JSON."""
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


def load_complaints() -> tuple[list[str], list[dict], list[str]]:
    """Load customer complaints (internal data)."""
    documents, metadatas, ids = [], [], []
    with open(DATA_DIR / "complaints.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (
                f"Complaint {row['id']} ({row['date']}): "
                f"Customer: {row['customer_name']} - Vehicle: {row['vehicle']} - "
                f"Category: {row['category']} - "
                f"Issue: {row['description']} - "
                f"Status: {row['status']} - Resolution: {row['resolution']}"
            )
            metadata = {
                "source": "complaints",
                "complaint_id": row["id"],
                "category": row["category"],
                "status": row["status"],
                "access_level": "internal",
            }
            documents.append(text)
            metadatas.append(metadata)
            ids.append(f"cmp_{i:03d}")
    return documents, metadatas, ids


# ---------------------------------------------------------------------------
# Collection setup
# ---------------------------------------------------------------------------

def build_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Build ChromaDB collection with all dealership data."""
    try:
        client.delete_collection("dealership_rag")
    except Exception:
        pass

    collection = client.create_collection(name="dealership_rag")

    all_docs, all_metas, all_ids = [], [], []
    source_counts = {}

    for source_name, loader_fn in [
        ("inventory", load_inventory),
        ("policies", load_policies),
        ("promotions", load_promotions),
        ("faq", load_faq),
        ("complaints", load_complaints),
    ]:
        docs, metas, ids = loader_fn()
        all_docs.extend(docs)
        all_metas.extend(metas)
        all_ids.extend(ids)
        source_counts[source_name] = len(docs)

    collection.add(documents=all_docs, metadatas=all_metas, ids=all_ids)

    print("Knowledge base loaded:")
    for source, count in source_counts.items():
        access = "internal" if source == "complaints" else "external"
        print(f"  {source:<15} {count:>3} documents  ({access})")
    external = sum(1 for m in all_metas if m.get("access_level") == "external")
    internal = sum(1 for m in all_metas if m.get("access_level") == "internal")
    print(f"  {'TOTAL':<15} {collection.count():>3} documents  "
          f"({external} external, {internal} internal)")
    print()

    return collection


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

def retrieve_documents(
    collection: chromadb.Collection,
    query: str,
    user_type: str,
    n_results: int = 5,
) -> dict:
    """
    Retrieve relevant documents from ChromaDB with access-level filtering.

    Args:
        collection: The ChromaDB collection to search.
        query: The user's question.
        user_type: Either "customer" (external only) or "employee" (all data).
        n_results: Number of documents to retrieve.

    Returns:
        ChromaDB query results dict with documents, metadatas, distances.
    """
    if user_type == "customer":
        # Customers can only see external data
        where_filter = {"access_level": "external"}
    else:
        # Employees can see everything -- no access_level filter needed
        where_filter = None

    query_kwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "distances", "metadatas"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)
    return results


def format_context(results: dict) -> str:
    """
    Format retrieved documents into a context string for the LLM.

    Includes source information and relevance scores to help the LLM
    understand the quality and origin of each piece of context.
    """
    if not results["documents"][0]:
        return "No relevant documents found."

    context_parts = []
    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ), 1):
        source = meta.get("source", "unknown")
        relevance = max(0, 1 - (dist / 2))  # Approximate relevance score 0-1
        context_parts.append(
            f"[Document {i}] (Source: {source}, Relevance: {relevance:.0%})\n{doc}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_system_prompt(user_type: str) -> str:
    """Get the system prompt based on user type."""
    base_prompt = (
        "You are a helpful assistant for AutoElite Motors, a car dealership. "
        "Answer questions based ONLY on the provided context documents. "
        "If the context does not contain enough information to answer the question, "
        "say so honestly rather than making up information. "
        "When referencing specific vehicles, include the year, make, model, and price. "
        "When referencing policies, cite the specific policy details."
    )

    if user_type == "customer":
        return (
            f"{base_prompt}\n\n"
            "You are speaking with a customer. Be friendly, professional, and helpful. "
            "Focus on available vehicles, current promotions, and dealership policies. "
            "Encourage them to visit the dealership or schedule a test drive when appropriate."
        )
    else:
        return (
            f"{base_prompt}\n\n"
            "You are speaking with a dealership employee. You have access to internal "
            "data including customer complaints and their resolution status. "
            "Be direct and detailed. Include complaint IDs, customer names, and "
            "resolution details when relevant."
        )


def rag_query(
    collection: chromadb.Collection,
    llm: ChatTogether,
    query: str,
    user_type: str,
    n_results: int = 5,
) -> tuple[str, dict]:
    """
    Execute a full RAG query: retrieve documents, format context, generate answer.

    Args:
        collection: ChromaDB collection.
        llm: The ChatTogether LLM instance.
        query: User's question.
        user_type: "customer" or "employee".
        n_results: Number of documents to retrieve.

    Returns:
        Tuple of (answer_text, retrieval_results).
    """
    # Step 1: Retrieve relevant documents
    results = retrieve_documents(collection, query, user_type, n_results)

    # Step 2: Format context
    context = format_context(results)

    # Step 3: Build the prompt
    system_prompt = get_system_prompt(user_type)
    user_prompt = (
        f"Context documents:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}"
    )

    # Step 4: Generate answer
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    answer = response.content

    return answer, results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_retrieved_docs(results: dict):
    """Display retrieved documents with their scores."""
    print("\n  Retrieved Documents:")
    print("  " + "-" * 76)

    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ), 1):
        source = meta.get("source", "unknown")
        access = meta.get("access_level", "?")
        relevance = max(0, 1 - (dist / 2))
        truncated = doc[:100].replace("\n", " ")
        if len(doc) > 100:
            truncated += "..."

        print(f"  [{i}] [{source}|{access}] (distance: {dist:.4f}, relevance: {relevance:.0%})")
        print(f"      {truncated}")
    print()


def display_answer(answer: str):
    """Display the LLM-generated answer."""
    print("  Answer:")
    print("  " + "-" * 76)
    # Wrap long lines for readability
    for line in answer.split("\n"):
        if len(line) > 78:
            words = line.split()
            current_line = " "
            for word in words:
                if len(current_line) + len(word) + 1 > 78:
                    print(f"  {current_line}")
                    current_line = "  " + word
                else:
                    current_line += " " + word
            if current_line.strip():
                print(f"  {current_line}")
        else:
            print(f"  {line}")
    print()


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def run_interactive_cli(collection: chromadb.Collection, llm: ChatTogether):
    """Run an interactive CLI for RAG queries."""
    print("=" * 80)
    print("AutoElite Motors -- RAG-Powered Assistant")
    print("=" * 80)
    print()

    # Select initial mode
    user_type = select_mode()

    print("\nCommands:")
    print('  Type a question to ask the assistant.')
    print('  Type "switch" to change between customer and employee mode.')
    print('  Type "quit" to exit.')
    print()

    while True:
        print("-" * 80)
        mode_label = "CUSTOMER" if user_type == "customer" else "EMPLOYEE"
        try:
            query = input(f"[{mode_label}] Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "quit":
            print("\nGoodbye!")
            break

        if query.lower() == "switch":
            user_type = select_mode()
            continue

        # Execute RAG query
        try:
            print(f"\n  Searching knowledge base as {mode_label.lower()}...")
            answer, results = rag_query(collection, llm, query, user_type)
            display_retrieved_docs(results)
            display_answer(answer)
        except Exception as e:
            print(f"\n  Error: {e}")
            print("  Please check your TOGETHER_API_KEY and try again.\n")


def select_mode() -> str:
    """Prompt the user to select customer or employee mode."""
    print("Select mode:")
    print("  1. Customer (sees only public information)")
    print("  2. Employee (sees all data including internal complaints)")
    print()

    while True:
        try:
            choice = input("Enter 1 or 2: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "customer"

        if choice == "1":
            print("\n  Mode set to CUSTOMER. You will see only external data.")
            return "customer"
        elif choice == "2":
            print("\n  Mode set to EMPLOYEE. You will see all data including internal records.")
            return "employee"
        else:
            print("  Please enter 1 or 2.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point."""
    print("Phase 3 - Complete RAG with Access-Controlled Retrieval")
    print("=" * 80)
    print()

    # Check for API key
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY environment variable is not set.")
        print()
        print("Set it in one of these ways:")
        print("  1. Create a .env file with: TOGETHER_API_KEY=your-key-here")
        print("  2. Export it: export TOGETHER_API_KEY=your-key-here")
        print()
        print("Get a free API key at: https://api.together.xyz/")
        return

    # Get model from environment or use default
    model = os.getenv("TOGETHER_MODEL", DEFAULT_MODEL)
    print(f"LLM Model: {model}")
    print()

    # Initialize LLM
    llm = ChatTogether(
        model=model,
        together_api_key=api_key,
        temperature=0.3,
        max_tokens=1024,
    )

    # Build knowledge base
    print("Loading dealership knowledge base...\n")
    client = chromadb.Client()
    collection = build_collection(client)

    # Run example queries first to show the system works
    print("=" * 80)
    print("EXAMPLE QUERIES (demonstrating both modes)")
    print("=" * 80)
    print()

    # Example 1: Customer query
    print("--- Example 1: Customer asking about SUVs ---")
    query1 = "What SUVs do you have available under $40,000?"
    print(f'Query: "{query1}"')
    print("Mode: customer\n")
    try:
        answer1, results1 = rag_query(collection, llm, query1, "customer")
        display_retrieved_docs(results1)
        display_answer(answer1)
    except Exception as e:
        print(f"  Error generating answer: {e}\n")

    # Example 2: Employee query
    print("--- Example 2: Employee asking about complaints ---")
    query2 = "What open complaints do we have about service issues?"
    print(f'Query: "{query2}"')
    print("Mode: employee\n")
    try:
        answer2, results2 = rag_query(collection, llm, query2, "employee")
        display_retrieved_docs(results2)
        display_answer(answer2)
    except Exception as e:
        print(f"  Error generating answer: {e}\n")

    # Start interactive mode
    print("\n" + "=" * 80)
    print("Entering interactive mode...")
    print("=" * 80 + "\n")
    run_interactive_cli(collection, llm)


if __name__ == "__main__":
    main()
