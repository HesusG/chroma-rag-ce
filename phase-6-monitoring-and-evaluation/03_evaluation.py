"""
Phase 6, Script 3: Evaluating RAG Quality

This script demonstrates how to evaluate the quality of a RAG pipeline.
It runs a set of test questions through the pipeline and measures:
  - Retrieval relevance: Are the retrieved documents relevant to the question?
  - Answer quality: Does the answer actually address the question?
  - Latency: How long does each step take?

Evaluation differs from monitoring: monitoring tells you what happened,
evaluation tells you whether what happened was good. This script uses two
complementary evaluation strategies:

  1. Source-based retrieval scoring: For each test case, we know which data
     sources (inventory, policies, etc.) should contain the answer. We check
     what fraction of retrieved documents come from those expected sources.
     This is a simple but effective proxy for retrieval precision.

  2. LLM-as-judge answer scoring: We use the LLM itself to judge whether
     the generated answer addresses the expected topic. This is a widely
     used technique for automated quality checks. The judge is asked a
     simple yes/no question ("Does this answer address the expected topic?")
     and responds with PASS or FAIL plus a brief explanation.

The results are printed as a formatted table and saved to a JSON file
for further analysis or comparison across runs.

Requirements:
  - TOGETHER_API_KEY set in .env
  - Data files in data/ directory

Usage:
  python 03_evaluation.py
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# -------------------------------------------------------------------------
# Test Cases
# -------------------------------------------------------------------------
# Each test case defines:
#   - question: The user query to evaluate
#   - expected_topic: A short description of what a good answer should cover
#   - expected_sources: Which data sources (by doc_type) should be retrieved
#
# The expected_sources use doc_type labels (not filenames) because that is
# what we store in ChromaDB metadata. This lets us check whether the
# retriever is pulling from the right category of documents.

TEST_CASES = [
    {
        "question": "What SUVs are available under $40,000?",
        "expected_topic": "SUV inventory with pricing",
        "expected_sources": ["inventory"],
    },
    {
        "question": "What is the return policy?",
        "expected_topic": "7-day return policy details",
        "expected_sources": ["policies"],
    },
    {
        "question": "Tell me about current promotions",
        "expected_topic": "active promotions and deals",
        "expected_sources": ["promotions"],
    },
    {
        "question": "What financing options do you offer?",
        "expected_topic": "financing terms and rates",
        "expected_sources": ["policies", "faq"],
    },
    {
        "question": "Do you have any electric vehicles?",
        "expected_topic": "EV inventory",
        "expected_sources": ["inventory"],
    },
    {
        "question": "How do I schedule a test drive?",
        "expected_topic": "test drive scheduling process",
        "expected_sources": ["faq", "policies"],
    },
    {
        "question": "What is covered under the warranty?",
        "expected_topic": "warranty coverage details",
        "expected_sources": ["policies"],
    },
    {
        "question": "Do you offer trade-in services?",
        "expected_topic": "trade-in process and valuation",
        "expected_sources": ["policies", "faq"],
    },
]


# -------------------------------------------------------------------------
# Phoenix Setup
# -------------------------------------------------------------------------

def setup_phoenix():
    """
    Launch Phoenix and instrument LangChain for tracing.

    This is the same setup used in Scripts 1 and 2. Every retrieval and LLM
    call during the evaluation will be traced in the Phoenix dashboard, so
    you can inspect individual test cases after the evaluation completes.

    Returns:
        The Phoenix session, or None if Phoenix is not available.
    """
    try:
        import phoenix as px

        session = px.launch_app()
        print(f"  Phoenix dashboard: {session.url}")

        from phoenix.otel import register
        tracer_provider = register(project_name="autoelite-rag")

        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        print("  LangChain instrumentation active.")
        return session

    except ImportError:
        print("  Phoenix not available. Running evaluation without tracing.")
        print("  Install with: pip install arize-phoenix openinference-instrumentation-langchain arize-phoenix-otel")
        return None

    except Exception as e:
        print(f"  Phoenix setup failed: {e}. Running without tracing.")
        return None


# -------------------------------------------------------------------------
# Document Loading
# -------------------------------------------------------------------------

def load_all_documents() -> list[dict]:
    """
    Load all dealership documents from the data/ directory.

    Each document is returned as a dict with:
      - text: The document content as a string
      - source: The filename it came from (e.g., "inventory.csv")
      - doc_type: A category label used for evaluation (inventory, policies,
        promotions, faq). These labels match the expected_sources in TEST_CASES.

    This is the same loading logic used in Script 2, ensuring consistency
    across the evaluation pipeline.
    """
    data_dir = Path(__file__).parent / "data"
    documents = []

    # --- Load inventory.csv ---
    # Each row becomes a single document describing one vehicle with all
    # its attributes (year, make, model, price, features, etc.).
    inventory_path = data_dir / "inventory.csv"
    if inventory_path.exists():
        with open(inventory_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (
                    f"{row['year']} {row['make']} {row['model']} {row['trim']} - "
                    f"${row['price']} - {row['condition']} - {row['color']} - "
                    f"Category: {row['category']} - Mileage: {row['mileage']} miles - "
                    f"Features: {row['features']} - Status: {row['status']}"
                )
                documents.append({
                    "text": text,
                    "source": "inventory.csv",
                    "doc_type": "inventory",
                })
        count = sum(1 for d in documents if d["source"] == "inventory.csv")
        print(f"  Loaded {count} inventory records.")

    # --- Load policies.txt ---
    # Split into sections on lines of dashes (the document uses dashed lines
    # as section dividers between different policy topics).
    policies_path = data_dir / "policies.txt"
    if policies_path.exists():
        content = policies_path.read_text()
        sections = []
        current_section = []
        for line in content.split("\n"):
            if (line.strip()
                    and all(c in "-" for c in line.strip())
                    and len(line.strip()) > 3):
                # This is a section divider -- save the previous section.
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text and not all(c in "= \n" for c in section_text):
                        sections.append(section_text)
                current_section = []
            else:
                current_section.append(line)
        # Do not forget the last section after the final divider.
        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text and not all(c in "= \n" for c in section_text):
                sections.append(section_text)

        for section in sections:
            documents.append({
                "text": section,
                "source": "policies.txt",
                "doc_type": "policies",
            })
        print(f"  Loaded {len(sections)} policy sections.")

    # --- Load promotions.md ---
    # Split on ## headers; each promotion section becomes a separate document.
    promotions_path = data_dir / "promotions.md"
    if promotions_path.exists():
        content = promotions_path.read_text()
        sections = content.split("\n## ")
        for i, section in enumerate(sections):
            text = section.strip()
            if text and not text.startswith("# AutoElite"):
                if not text.startswith("## "):
                    text = "## " + text
                documents.append({
                    "text": text,
                    "source": "promotions.md",
                    "doc_type": "promotions",
                })
        count = sum(1 for d in documents if d["source"] == "promotions.md")
        print(f"  Loaded {count} promotions.")

    # --- Load faq.json ---
    # Each FAQ item becomes a "Q: ... A: ..." document.
    faq_path = data_dir / "faq.json"
    if faq_path.exists():
        with open(faq_path, "r") as f:
            faq_data = json.load(f)
        for item in faq_data:
            text = f"Q: {item['question']}\nA: {item['answer']}"
            documents.append({
                "text": text,
                "source": "faq.json",
                "doc_type": "faq",
            })
        print(f"  Loaded {len(faq_data)} FAQ entries.")

    print(f"  Total documents: {len(documents)}")
    return documents


# -------------------------------------------------------------------------
# ChromaDB + LLM Setup
# -------------------------------------------------------------------------

def build_rag_chain(documents: list[dict]):
    """
    Build a RAG pipeline using ChromaDB directly (not the LangChain Chroma
    wrapper) to avoid embedding dependency issues with TogetherEmbeddings.

    ChromaDB's default embedding function uses a local Sentence Transformers
    model (all-MiniLM-L6-v2), which runs entirely on your machine. This is
    more reliable for evaluation than an API-based embedder that may hit rate
    limits during a batch of test cases.

    Returns:
        A tuple of (chromadb_collection, llm_instance) so we can use them
        separately for retrieval evaluation and answer evaluation.
    """
    import chromadb
    from langchain_together import ChatTogether

    # --- Set up ChromaDB with its default embedding function ---
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="autoelite_evaluation",
    )

    # Only add documents if the collection is empty (avoids duplicates on rerun).
    if collection.count() == 0:
        texts = [doc["text"] for doc in documents]
        metadatas = [
            {"source": doc["source"], "doc_type": doc["doc_type"]}
            for doc in documents
        ]
        ids = [f"doc_{i}" for i in range(len(documents))]

        collection.add(documents=texts, metadatas=metadatas, ids=ids)
        print(f"  Added {len(texts)} documents to ChromaDB collection.")
    else:
        print(f"  Collection already has {collection.count()} documents.")

    # --- Set up the LLM ---
    llm = ChatTogether(
        model=os.getenv(
            "TOGETHER_MODEL",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ),
        temperature=0.3,
        max_tokens=1024,
    )

    return collection, llm


# -------------------------------------------------------------------------
# Retrieval Evaluation
# -------------------------------------------------------------------------

def evaluate_retrieval(collection, test_cases: list[dict], n_results: int = 4) -> list[dict]:
    """
    Evaluate retrieval quality for each test case.

    For each question, we:
      1. Query ChromaDB to get the top-k documents
      2. Check how many of the retrieved documents come from an expected source
      3. Compute a precision score: (relevant retrieved docs) / (total retrieved docs)

    A "relevant" document is one whose doc_type matches any of the test case's
    expected_sources. For example, if the expected sources are ["inventory"],
    then a retrieved document with doc_type "inventory" counts as relevant,
    while one with doc_type "faq" does not.

    This is a simple proxy for retrieval relevance. In a production system you
    would use more sophisticated metrics like NDCG or an LLM-based relevance
    judge per document.

    Returns:
        A list of result dicts, one per test case, with retrieval scores
        and the retrieved documents for inspection.
    """
    results = []

    for test_case in test_cases:
        question = test_case["question"]
        expected_sources = test_case["expected_sources"]

        # Query ChromaDB for the top-k results
        query_results = collection.query(
            query_texts=[question],
            n_results=n_results,
        )

        # Extract documents with their metadata and distances
        retrieved_docs = []
        relevant_count = 0

        if query_results and query_results["documents"]:
            docs = query_results["documents"][0]
            metadatas = query_results["metadatas"][0]
            distances = query_results["distances"][0]

            for doc_text, meta, dist in zip(docs, metadatas, distances):
                doc_type = meta.get("doc_type", "unknown")
                is_relevant = doc_type in expected_sources

                if is_relevant:
                    relevant_count += 1

                retrieved_docs.append({
                    "text": doc_text[:150],  # Truncate for readability
                    "source": meta.get("source", "unknown"),
                    "doc_type": doc_type,
                    "distance": round(dist, 4),
                    "is_relevant": is_relevant,
                })

        # Retrieval precision: what fraction of retrieved docs are relevant?
        total_retrieved = len(retrieved_docs)
        precision = relevant_count / total_retrieved if total_retrieved > 0 else 0.0

        results.append({
            "question": question,
            "expected_sources": expected_sources,
            "retrieved_docs": retrieved_docs,
            "relevant_count": relevant_count,
            "total_retrieved": total_retrieved,
            "retrieval_precision": round(precision, 4),
        })

    return results


# -------------------------------------------------------------------------
# Answer Evaluation (LLM-as-Judge)
# -------------------------------------------------------------------------

def evaluate_answer(collection, llm, test_cases: list[dict], n_results: int = 4) -> list[dict]:
    """
    Evaluate answer quality by running the full RAG pipeline for each test case.

    For each question, we:
      1. Retrieve context from ChromaDB
      2. Build a prompt with the retrieved context and the question
      3. Call the LLM to generate an answer
      4. Use the LLM again as a "judge" to assess whether the answer
         addresses the expected topic
      5. Measure end-to-end latency (retrieval + generation + judging)

    The "LLM-as-judge" pattern is a widely used technique for automated
    evaluation. We ask the LLM a simple yes/no question: "Does this answer
    address the expected topic?" This is not perfect -- the judge model can
    make mistakes, and it may be biased toward its own outputs -- but it
    scales much better than human evaluation and catches many obvious failures
    like hallucinations, off-topic answers, or empty responses.

    Returns:
        A list of result dicts with answers, judgments, and latency.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    results = []

    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        expected_topic = test_case["expected_topic"]

        print(f"  [{i + 1}/{len(test_cases)}] Evaluating: \"{question}\"")

        start_time = time.time()

        # --- Step 1: Retrieve context ---
        query_results = collection.query(
            query_texts=[question],
            n_results=n_results,
        )
        retrieval_time = time.time() - start_time

        # --- Step 2: Build the RAG prompt ---
        context_parts = []
        if query_results and query_results["documents"]:
            for j, doc in enumerate(query_results["documents"][0], 1):
                source = query_results["metadatas"][0][j - 1].get("source", "unknown")
                context_parts.append(f"[Document {j} - {source}]\n{doc}")

        context = "\n\n".join(context_parts) if context_parts else "No documents found."

        rag_messages = [
            SystemMessage(
                content=(
                    "You are a helpful sales assistant for AutoElite Motors. "
                    "Answer the customer's question using ONLY the information "
                    "provided in the context below. If the context does not "
                    "contain enough information, say so honestly -- do not "
                    "make up information.\n\n"
                    f"Context:\n{context}"
                )
            ),
            HumanMessage(content=question),
        ]

        # --- Step 3: Generate the answer ---
        try:
            response = llm.invoke(rag_messages)
            answer = response.content
            generation_time = time.time() - start_time - retrieval_time
        except Exception as e:
            answer = f"Error generating answer: {e}"
            generation_time = 0.0

        # --- Step 4: LLM-as-Judge ---
        # Ask the LLM to evaluate whether the answer addresses the topic.
        # We use a strict response format (PASS/FAIL on the first line) to
        # make parsing reliable.
        try:
            judge_messages = [
                SystemMessage(
                    content=(
                        "You are an evaluation judge. Your job is to determine "
                        "if an answer adequately addresses the expected topic. "
                        "Respond with ONLY 'PASS' or 'FAIL' on the first line, "
                        "followed by a brief one-sentence explanation on the "
                        "second line."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {question}\n"
                        f"Expected topic: {expected_topic}\n"
                        f"Answer provided:\n{answer}\n\n"
                        "Does the answer address the expected topic? "
                        "Reply PASS if it does, FAIL if it does not."
                    )
                ),
            ]

            judge_response = llm.invoke(judge_messages)
            judge_text = judge_response.content.strip()

            # Parse the judgment -- look for PASS or FAIL at the start.
            first_line = judge_text.split("\n")[0].strip().upper()
            if "PASS" in first_line:
                judgment = "Pass"
            elif "FAIL" in first_line:
                judgment = "Fail"
            else:
                # Fallback: check the full response text for keywords.
                judgment = "Pass" if "PASS" in judge_text.upper() else "Fail"

            # Extract the explanation (second line if available).
            judge_explanation = (
                judge_text.split("\n")[1].strip()
                if "\n" in judge_text
                else judge_text
            )

        except Exception as e:
            judgment = "Error"
            judge_explanation = str(e)

        total_latency = time.time() - start_time

        results.append({
            "question": question,
            "expected_topic": expected_topic,
            "answer": answer,
            "judgment": judgment,
            "judge_explanation": judge_explanation,
            "retrieval_latency_s": round(retrieval_time, 3),
            "generation_latency_s": round(generation_time, 3),
            "total_latency_s": round(total_latency, 3),
        })

        # Print a brief status indicator for progress tracking
        status = "OK" if judgment == "Pass" else "XX"
        print(f"           [{status}] {judgment} ({total_latency:.1f}s)")

    return results


# -------------------------------------------------------------------------
# Report Generation
# -------------------------------------------------------------------------

def print_evaluation_report(retrieval_results: list[dict], answer_results: list[dict]):
    """
    Print a formatted evaluation report to the console.

    The report shows a summary table with one row per test case, followed
    by aggregate statistics. This format makes it easy to scan for problem
    areas: low retrieval precision indicates the retriever is returning
    wrong documents, a "Fail" judgment indicates the LLM answer missed the
    topic, and high latency indicates performance issues.
    """
    print("\n")
    print("=" * 76)
    print("  EVALUATION REPORT")
    print("=" * 76)

    # Table header
    header = f"  {'Question':<38} | {'Retrieval':>10} | {'Answer':>7} | {'Latency':>8}"
    print(header)
    print("  " + "-" * 38 + "-+-" + "-" * 10 + "-+-" + "-" * 7 + "-+-" + "-" * 8)

    # Table rows -- one per test case
    total_precision = 0.0
    pass_count = 0
    total_latency = 0.0

    for ret, ans in zip(retrieval_results, answer_results):
        # Truncate question for display
        question = ret["question"]
        if len(question) > 36:
            question = question[:33] + "..."

        precision = ret["retrieval_precision"]
        judgment = ans["judgment"]
        latency = ans["total_latency_s"]

        total_precision += precision
        if judgment == "Pass":
            pass_count += 1
        total_latency += latency

        precision_str = f"{precision * 100:.0f}%"
        latency_str = f"{latency:.1f}s"

        print(f"  {question:<38} | {precision_str:>10} | {judgment:>7} | {latency_str:>8}")

    # Separator before summary
    print("  " + "-" * 38 + "-+-" + "-" * 10 + "-+-" + "-" * 7 + "-+-" + "-" * 8)

    # Aggregate statistics
    n = len(retrieval_results)
    avg_precision = total_precision / n if n > 0 else 0.0
    pass_rate = pass_count / n if n > 0 else 0.0
    avg_latency = total_latency / n if n > 0 else 0.0

    print()
    print(f"  Average Retrieval Precision: {avg_precision * 100:.1f}%")
    print(f"  Answer Pass Rate:            {pass_rate * 100:.1f}% ({pass_count}/{n})")
    print(f"  Average Latency:             {avg_latency:.1f}s")
    print()
    print("=" * 76)


def save_results(retrieval_results: list[dict], answer_results: list[dict], output_path: Path):
    """
    Save the full evaluation results to a JSON file.

    The JSON file contains all retrieved documents, answers, judgments,
    and scores for every test case. This is useful for:
      - Detailed analysis of failure cases
      - Comparing results across runs (after changing prompts, models, etc.)
      - Feeding into a dashboard or reporting tool
    """
    combined = []
    for ret, ans in zip(retrieval_results, answer_results):
        combined.append({
            "question": ret["question"],
            "expected_sources": ret["expected_sources"],
            "expected_topic": ans["expected_topic"],
            "retrieval": {
                "precision": ret["retrieval_precision"],
                "relevant_count": ret["relevant_count"],
                "total_retrieved": ret["total_retrieved"],
                "documents": ret["retrieved_docs"],
            },
            "answer": {
                "text": ans["answer"],
                "judgment": ans["judgment"],
                "judge_explanation": ans["judge_explanation"],
            },
            "latency": {
                "retrieval_s": ans["retrieval_latency_s"],
                "generation_s": ans["generation_latency_s"],
                "total_s": ans["total_latency_s"],
            },
        })

    output = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_test_cases": len(combined),
        "results": combined,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to: {output_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6: Evaluating RAG Quality")
    print("=" * 60)

    # Check for API key
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

    # Step 1: Launch Phoenix (optional, for trace visibility during evaluation)
    print("\n[Step 1] Setting up Phoenix (optional)...")
    session = setup_phoenix()

    # Step 2: Load documents
    print("\n[Step 2] Loading documents...")
    documents = load_all_documents()

    if not documents:
        print("  ERROR: No documents loaded. Check the data/ directory.")
        sys.exit(1)

    # Step 3: Build the RAG components
    print("\n[Step 3] Building RAG pipeline...")
    collection, llm = build_rag_chain(documents)

    # Step 4: Evaluate retrieval quality
    print("\n[Step 4] Evaluating retrieval quality...")
    retrieval_results = evaluate_retrieval(collection, TEST_CASES)

    # Step 5: Evaluate answer quality (this calls the LLM twice per test case)
    print("\n[Step 5] Evaluating answer quality (this may take a minute)...")
    answer_results = evaluate_answer(collection, llm, TEST_CASES)

    # Step 6: Print the formatted report
    print_evaluation_report(retrieval_results, answer_results)

    # Step 7: Save detailed results to JSON
    output_path = Path(__file__).parent / "evaluation_results.json"
    print(f"\n[Step 7] Saving detailed results...")
    save_results(retrieval_results, answer_results, output_path)

    # Final message with Phoenix dashboard link if available
    if session:
        print(f"\n  Phoenix dashboard: {session.url}")
        print("  Check the dashboard to inspect individual evaluation traces.")
        try:
            input("\nPress Enter to exit...")
        except KeyboardInterrupt:
            pass
        print("\nPhoenix server stopped. Goodbye.")
    else:
        print("\nEvaluation complete.")
