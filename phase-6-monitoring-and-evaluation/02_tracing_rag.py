"""
Phase 6, Script 2: Tracing a Full RAG Pipeline with Phoenix

This script builds a complete RAG pipeline using LangChain and traces every
step with Arize Phoenix. Unlike Script 1 which traced a single LLM call, this
script traces the full pipeline:

  1. Document retrieval (embedding the query + searching ChromaDB)
  2. Context formatting (building the prompt from retrieved documents)
  3. LLM generation (producing the final answer)

Each of these steps appears as a separate span in Phoenix, nested under a
single trace for each query. This lets you see exactly what the retriever
found, what prompt was sent to the LLM, and how long each step took.

We use LangChain's Chroma integration and LCEL (LangChain Expression Language)
to build the chain, because LangChain chains are automatically traced by the
OpenInference instrumentor.

After running, open http://localhost:6006 to explore the traces.

Requirements:
  - TOGETHER_API_KEY set in .env file or as an environment variable
  - Data files in the data/ directory

Usage:
  python 02_tracing_rag.py
"""

import os
import sys
import csv
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# -------------------------------------------------------------------------
# Phoenix Setup (same pattern as Script 1)
# -------------------------------------------------------------------------

def setup_phoenix():
    """Launch Phoenix and instrument LangChain for tracing."""
    import phoenix as px

    session = px.launch_app()
    print(f"  Phoenix dashboard: {session.url}")

    from phoenix.otel import register
    tracer_provider = register(project_name="autoelite-rag")

    from openinference.instrumentation.langchain import LangChainInstrumentor
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    print("  LangChain instrumentation active.")
    return session


# -------------------------------------------------------------------------
# Document Loading
# -------------------------------------------------------------------------

def load_all_documents() -> list[dict]:
    """
    Load all dealership documents from the data/ directory.

    Returns a list of dicts, each with 'text', 'source', and 'doc_type' keys.
    This is the same document loading approach used in earlier phases, but
    structured for easy ingestion into LangChain's Chroma vectorstore.
    """
    data_dir = Path(__file__).parent / "data"
    documents = []

    # --- Load inventory.csv ---
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
        print(f"  Loaded {sum(1 for d in documents if d['source'] == 'inventory.csv')} inventory records.")

    # --- Load policies.txt ---
    policies_path = data_dir / "policies.txt"
    if policies_path.exists():
        content = policies_path.read_text()
        # Split by section headers (lines of dashes)
        sections = []
        current_section = []
        for line in content.split("\n"):
            if line.strip() and all(c in "-" for c in line.strip()) and len(line.strip()) > 3:
                # This is a section divider. Save the previous section.
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text and not all(c in "= \n" for c in section_text):
                        sections.append(section_text)
                current_section = []
            else:
                current_section.append(line)
        # Do not forget the last section
        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text and not all(c in "= \n" for c in section_text):
                sections.append(section_text)

        for section in sections:
            documents.append({
                "text": section,
                "source": "policies.txt",
                "doc_type": "policy",
            })
        print(f"  Loaded {len(sections)} policy sections.")

    # --- Load promotions.md ---
    promotions_path = data_dir / "promotions.md"
    if promotions_path.exists():
        content = promotions_path.read_text()
        # Split by ## headers
        sections = content.split("\n## ")
        for i, section in enumerate(sections):
            text = section.strip()
            if text and not text.startswith("# AutoElite"):
                if not text.startswith("## "):
                    text = "## " + text
                documents.append({
                    "text": text,
                    "source": "promotions.md",
                    "doc_type": "promotion",
                })
        print(f"  Loaded {sum(1 for d in documents if d['source'] == 'promotions.md')} promotions.")

    # --- Load faq.json ---
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
# ChromaDB + LangChain Chain Setup
# -------------------------------------------------------------------------

def build_rag_chain(documents: list[dict]):
    """
    Build a LangChain RAG chain with Chroma as the vectorstore.

    This uses LangChain's high-level abstractions so that Phoenix can
    automatically trace each step:
      - Chroma vectorstore (traced as a retriever span)
      - ChatPromptTemplate (traced as part of the chain)
      - ChatTogether LLM (traced as an LLM span)

    Returns:
        A tuple of (chain, retriever) so we can use both.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_together import ChatTogether, TogetherEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.documents import Document

    # Convert our document dicts to LangChain Document objects
    langchain_docs = [
        Document(
            page_content=doc["text"],
            metadata={"source": doc["source"], "doc_type": doc["doc_type"]},
        )
        for doc in documents
    ]

    # Create a Chroma vectorstore from the documents.
    # Using TogetherEmbeddings so the embedding step is also traced.
    print("\n  Creating ChromaDB vectorstore with TogetherEmbeddings...")
    embeddings = TogetherEmbeddings(
        model="togethercomputer/m2-bert-80M-8k-retrieval",
    )
    vectorstore = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        collection_name="autoelite_traced",
    )
    print(f"  Vectorstore created with {len(langchain_docs)} documents.")

    # Create a retriever from the vectorstore.
    # k=4 means retrieve the top 4 most relevant documents.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Define the prompt template.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful sales assistant for AutoElite Motors, a car dealership. "
            "Answer the customer's question using ONLY the information provided in the "
            "context below. If the context does not contain enough information to answer "
            "the question, say so honestly -- do not make up information.\n\n"
            "Be friendly, concise, and helpful. If the customer asks about a specific "
            "vehicle, include relevant details like price, features, and availability.\n\n"
            "Context:\n{context}"
        ),
        ("human", "{question}"),
    ])

    # Initialize the LLM.
    llm = ChatTogether(
        model=os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        temperature=0.3,
        max_tokens=1024,
    )

    # Helper to format retrieved documents into a single string.
    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[Document {i} - {source}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    # Build the LCEL chain.
    # RunnablePassthrough passes the question through unchanged.
    # The retriever fetches documents, format_docs converts them to a string.
    # The prompt template fills in the context and question.
    # The LLM generates the response.
    # StrOutputParser extracts the text content from the LLM response.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# -------------------------------------------------------------------------
# Test Queries
# -------------------------------------------------------------------------

def run_test_queries(chain):
    """
    Run several test queries through the RAG chain.

    Each query generates a full trace in Phoenix showing:
      - The retriever span (which documents were found)
      - The LLM span (the full prompt and response)
      - Latency and token usage for each step
    """
    test_queries = [
        "What SUVs do you have available under $40,000?",
        "What is your return policy?",
        "Tell me about current promotions and deals.",
        "Do you have any electric vehicles?",
        "What financing options are available for first-time buyers?",
    ]

    print("\n  Running test queries through the traced RAG pipeline...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"  [{i}/{len(test_queries)}] Query: \"{query}\"")

        try:
            # Invoke the chain. This triggers:
            #   1. Retriever: embeds the query, searches ChromaDB
            #   2. format_docs: formats retrieved documents
            #   3. Prompt: builds the full prompt
            #   4. LLM: generates the response
            # All steps are automatically traced by Phoenix.
            response = chain.invoke(query)

            # Print a preview of the response
            preview = response[:200].replace("\n", " ")
            if len(response) > 200:
                preview += "..."
            print(f"  Response: {preview}")
            print()

        except Exception as e:
            print(f"  Error: {e}")
            print()

    return test_queries


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6: Tracing a Full RAG Pipeline")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print()
        print("  ERROR: TOGETHER_API_KEY not found.")
        print("  Set it in a .env file or export it in your terminal:")
        print()
        print('    export TOGETHER_API_KEY="your-key-here"')
        sys.exit(1)

    # Step 1: Launch Phoenix
    print("\n[Step 1] Launching Phoenix...")
    session = setup_phoenix()

    # Step 2: Load documents
    print("\n[Step 2] Loading documents...")
    documents = load_all_documents()

    # Step 3: Build the RAG chain
    print("\n[Step 3] Building the LangChain RAG chain...")
    chain, retriever = build_rag_chain(documents)

    # Step 4: Run test queries
    print("\n[Step 4] Running test queries...")
    test_queries = run_test_queries(chain)

    # Step 5: Prompt user to check the dashboard
    print("=" * 60)
    print(f"  All {len(test_queries)} queries complete.")
    print(f"  Open the Phoenix dashboard at: {session.url}")
    print()
    print("  In the dashboard you should see:")
    print(f"    - {len(test_queries)} traces (one per query)")
    print("    - Each trace contains retriever + LLM spans")
    print("    - Click a trace to see retrieved documents,")
    print("      the full prompt, token usage, and latency")
    print("=" * 60)

    # Keep running so the dashboard stays available
    try:
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        pass

    print("\nPhoenix server stopped. Goodbye.")
