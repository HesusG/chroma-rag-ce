"""
Phase 6, Script 4: Final Chatbot with Full Monitoring

The complete AutoElite Motors chatbot with:
  - All Phase 5 features (context engineering, memory, guardrails)
  - Arize Phoenix tracing (every query traced)
  - Token usage tracking
  - Latency tracking

This script brings together everything from the course:
  - Phase 2: Document loading and ChromaDB indexing
  - Phase 3: Retrieval with relevance filtering
  - Phase 4: Chainlit UI with streaming responses
  - Phase 5: Context engineering (prompts, memory, guardrails)
  - Phase 6: Observability with Phoenix tracing

Every user query generates a trace in Phoenix showing the retrieval step,
the LLM call, token counts, and latency. The response footer displays
sources, token usage, latency, and a link to the Phoenix dashboard.

Run with:
    chainlit run 04_chainlit_app.py

Open Phoenix dashboard at http://localhost:6006 for traces.

Requirements:
  - TOGETHER_API_KEY set in .env
  - Data files in data/ directory
"""

import os
import csv
import json
import time
from pathlib import Path
from dotenv import load_dotenv

import chromadb
import chainlit as cl
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()


# -------------------------------------------------------------------------
# Phoenix Setup
# -------------------------------------------------------------------------
# We initialize Phoenix at module load time so it is ready before any
# Chainlit request arrives. If Phoenix is not installed or fails to start,
# the chatbot still works -- it just runs without tracing.
#
# The setup_phoenix() function encapsulates all Phoenix initialization so
# it can be reused across scripts. It launches the Phoenix server, registers
# the OpenTelemetry tracer provider, and instruments LangChain.

def setup_phoenix():
    """Initialize Phoenix tracing. Returns session or None if unavailable."""
    try:
        import phoenix as px

        session = px.launch_app()

        from phoenix.otel import register
        tracer_provider = register(project_name="autoelite-rag")

        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        print(f"Phoenix dashboard: {session.url}")
        return session

    except ImportError:
        print("Phoenix not available. Running without tracing.")
        return None

    except Exception as e:
        print(f"Phoenix setup failed: {e}. Running without tracing.")
        return None


phoenix_session = setup_phoenix()


# -------------------------------------------------------------------------
# Token Counting
# -------------------------------------------------------------------------
# tiktoken provides accurate token counts for OpenAI-compatible tokenizers.
# The cl100k_base encoding is a reasonable approximation for most LLMs,
# including Llama models (which use a different tokenizer but have a similar
# token-to-word ratio). If tiktoken is not installed, we fall back to a
# rough word-count estimate.

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Count tokens using the cl100k_base encoding."""
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        """Approximate token count by splitting on whitespace."""
        return len(text.split())


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
MODELS = {
    "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "DeepSeek R1 70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Llama 3.1 8B": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
}
MAX_HISTORY = 10
RELEVANCE_THRESHOLD = 1.5  # ChromaDB L2 distance; lower = more relevant

# Phoenix dashboard URL (used in response footer links)
PHOENIX_URL = "http://localhost:6006"


# -------------------------------------------------------------------------
# System Prompts (XML-tagged, same as Phase 5)
# -------------------------------------------------------------------------
# XML tags make the prompt structure explicit for the LLM. Each section
# (<role>, <task>, <constraints>, <output_format>) tells the model exactly
# what is expected. This is a context engineering best practice from Phase 5.

CUSTOMER_PROMPT = """<role>
You are a friendly and knowledgeable sales assistant at AutoElite Motors.
</role>

<task>
Help customers find vehicles, understand financing, learn about promotions, and answer policy questions. Use the provided context for accurate responses.
</task>

<constraints>
- Only answer questions related to AutoElite Motors and automotive topics
- If asked about unrelated topics, politely redirect
- Never fabricate inventory, prices, or terms not in the context
- If unsure, suggest contacting the dealership directly
</constraints>

<output_format>
- Lead with a direct answer
- Use bullet points for multiple items
- Include specific details when available
- End with a helpful next step
- Keep responses concise (2-4 paragraphs)
</output_format>"""

EMPLOYEE_PROMPT = """<role>
You are an internal assistant for AutoElite Motors employees.
</role>

<task>
Help employees access complaints, support tickets, training procedures, inventory status, and internal policies. Provide data-driven, actionable responses.
</task>

<constraints>
- Provide factual, concise answers based on context
- Include relevant details (dates, status, categories) for complaints and tickets
- Flag urgent or escalated items
- If information is not in context, say so clearly
</constraints>

<output_format>
- Be concise and data-driven
- Use structured lists for multiple items
- Include status, dates, and categories when relevant
- Highlight action items needing attention
</output_format>"""


# -------------------------------------------------------------------------
# Data Loading Helpers
# -------------------------------------------------------------------------
# Each loader reads one file type and returns (docs, metadatas, ids) tuples
# that can be added directly to a ChromaDB collection. This is the same
# pattern used in Phase 5's chainlit app.

def load_inventory() -> tuple:
    """Load vehicle inventory from CSV. Each row becomes one document."""
    docs, metas, ids = [], [], []
    path = DATA_DIR / "inventory.csv"
    if not path.exists():
        return docs, metas, ids
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (
                f"{row['year']} {row['make']} {row['model']} {row['trim']}: "
                f"${row['price']}, {row['mileage']} miles, {row['condition']}, "
                f"{row['color']}, {row['category']}. "
                f"Features: {row['features']}. Status: {row['status']}."
            )
            docs.append(text)
            metas.append({"source": "inventory.csv", "category": row.get("category", "")})
            ids.append(f"inv_{i}")
    return docs, metas, ids


def load_complaints() -> tuple:
    """Load customer complaints from CSV (employee-only data)."""
    docs, metas, ids = [], [], []
    path = DATA_DIR / "complaints.csv"
    if not path.exists():
        return docs, metas, ids
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (
                f"Complaint {row['id']} ({row['date']}): {row['customer_name']} - "
                f"{row['vehicle']}. Category: {row['category']}. "
                f"Description: {row['description']} "
                f"Status: {row['status']}. Resolution: {row['resolution']}."
            )
            docs.append(text)
            metas.append({"source": "complaints.csv", "status": row.get("status", "")})
            ids.append(f"comp_{i}")
    return docs, metas, ids


def load_policies() -> tuple:
    """Load dealership policies from text file, split into sections."""
    docs, metas, ids = [], [], []
    path = DATA_DIR / "policies.txt"
    if not path.exists():
        return docs, metas, ids
    with open(path, "r") as f:
        content = f.read()

    # Split on double newlines and accumulate sections of meaningful size.
    # Short paragraphs are merged to avoid tiny documents with poor retrieval.
    sections = content.split("\n\n")
    current_section = ""
    section_idx = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue
        current_section += section + "\n\n"
        if len(current_section) > 200:
            docs.append(current_section.strip())
            metas.append({"source": "policies.txt"})
            ids.append(f"pol_{section_idx}")
            section_idx += 1
            current_section = ""
    if current_section.strip():
        docs.append(current_section.strip())
        metas.append({"source": "policies.txt"})
        ids.append(f"pol_{section_idx}")
    return docs, metas, ids


def load_promotions() -> tuple:
    """Load promotions from markdown file, split on ## headers."""
    docs, metas, ids = [], [], []
    path = DATA_DIR / "promotions.md"
    if not path.exists():
        return docs, metas, ids
    with open(path, "r") as f:
        content = f.read()

    sections = content.split("## ")
    for i, section in enumerate(sections):
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        docs.append(section)
        metas.append({"source": "promotions.md"})
        ids.append(f"promo_{i}")
    return docs, metas, ids


def load_faq() -> tuple:
    """Load FAQ entries from JSON file."""
    docs, metas, ids = [], [], []
    path = DATA_DIR / "faq.json"
    if not path.exists():
        return docs, metas, ids
    with open(path, "r") as f:
        faqs = json.load(f)
    for i, faq in enumerate(faqs):
        text = f"Q: {faq['question']}\nA: {faq['answer']}"
        docs.append(text)
        metas.append({"source": "faq.json", "category": faq.get("category", "")})
        ids.append(f"faq_{i}")
    return docs, metas, ids


def load_training_manual() -> tuple:
    """
    Load the training manual PDF, if pypdf is available.

    PDF loading requires the pypdf library. If it is not installed, we
    skip this file gracefully -- the chatbot still works with all other
    data sources. This is important because pypdf is an optional dependency
    that some users may not have installed.
    """
    docs, metas, ids = [], [], []
    path = DATA_DIR / "training_manual.pdf"
    if not path.exists():
        return docs, metas, ids

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(text.strip())
                metas.append({"source": "training_manual.pdf", "page": str(i + 1)})
                ids.append(f"manual_{i}")
    except ImportError:
        print("  pypdf not installed -- skipping training_manual.pdf")
    except Exception as e:
        print(f"  Could not load training_manual.pdf: {e}")

    return docs, metas, ids


# -------------------------------------------------------------------------
# ChromaDB Collection Setup
# -------------------------------------------------------------------------

def setup_collections():
    """
    Create and populate two ChromaDB collections:
      - customer_docs: inventory, policies, promotions, FAQ
      - employee_docs: everything above + complaints + training manual

    The customer collection excludes complaints (those are internal) and
    the training manual (employee-only). The employee collection includes
    everything for maximum coverage.

    Returns:
        Tuple of (customer_collection, employee_collection).
    """
    client = chromadb.Client()

    # --- Customer collection ---
    customer_col = client.get_or_create_collection("customer_docs")
    if customer_col.count() == 0:
        all_docs, all_metas, all_ids = [], [], []
        for loader in [load_inventory, load_policies, load_promotions, load_faq]:
            docs, metas, ids = loader()
            all_docs.extend(docs)
            all_metas.extend(metas)
            all_ids.extend(ids)
        if all_docs:
            customer_col.add(
                documents=all_docs, metadatas=all_metas, ids=all_ids
            )
        print(f"  Customer collection: {customer_col.count()} documents")

    # --- Employee collection (includes complaints and training manual) ---
    employee_col = client.get_or_create_collection("employee_docs")
    if employee_col.count() == 0:
        all_docs, all_metas, all_ids = [], [], []
        for loader in [
            load_inventory, load_complaints, load_policies,
            load_promotions, load_faq, load_training_manual,
        ]:
            docs, metas, ids = loader()
            all_docs.extend(docs)
            all_metas.extend(metas)
            all_ids.extend(ids)
        if all_docs:
            employee_col.add(
                documents=all_docs, metadatas=all_metas, ids=all_ids
            )
        print(f"  Employee collection: {employee_col.count()} documents")

    return customer_col, employee_col


# -------------------------------------------------------------------------
# Retrieval with Relevance Filtering
# -------------------------------------------------------------------------

def retrieve_relevant_docs(collection, query: str, n_results: int = 8) -> list:
    """
    Query ChromaDB and filter results by relevance threshold.

    ChromaDB returns L2 (Euclidean) distances by default. Lower distance
    means higher similarity. We discard documents above RELEVANCE_THRESHOLD
    to avoid injecting irrelevant context into the prompt -- this is better
    than blindly including all top-k results.

    Returns:
        A list of dicts, each with 'text', 'distance', and 'source' keys,
        sorted by relevance (lowest distance first).
    """
    results = collection.query(query_texts=[query], n_results=n_results)

    filtered_docs = []
    if results and results["documents"] and results["distances"]:
        docs = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for doc, distance, meta in zip(docs, distances, metadatas):
            if distance <= RELEVANCE_THRESHOLD:
                filtered_docs.append({
                    "text": doc,
                    "distance": distance,
                    "source": meta.get("source", "unknown"),
                })

    # Sort by relevance (lowest distance = most relevant)
    filtered_docs.sort(key=lambda x: x["distance"])
    return filtered_docs


# -------------------------------------------------------------------------
# Guardrails
# -------------------------------------------------------------------------

def check_guardrails(query: str, mode: str) -> str | None:
    """
    Check if the query needs special handling before going through RAG.

    Two guardrails are implemented:
      1. Complaint redirect (customer mode only): If a customer mentions
         complaints or frustration, redirect them to the support team
         rather than having the chatbot try to resolve complaints.
      2. Off-topic detection (all modes): If the query is clearly unrelated
         to automotive topics, politely redirect.

    Returns:
        A redirect message string if the guardrail triggers, or None if
        the query should proceed normally through the RAG pipeline.
    """
    query_lower = query.lower()

    # Guardrail 1: Complaint redirect for customer mode
    complaint_keywords = [
        "complaint", "complain", "unhappy", "frustrated",
        "terrible", "worst", "sue", "lawyer", "refund",
        "scam", "ripped off", "angry",
    ]
    if mode == "customer" and any(kw in query_lower for kw in complaint_keywords):
        return (
            "I understand you have a concern, and I am sorry to hear that. "
            "For complaints and escalations, I recommend contacting our "
            "Customer Support team directly:\n\n"
            "- **Phone:** (555) 123-4567\n"
            "- **Email:** support@autoelitemotors.com\n"
            "- **Hours:** Monday-Friday, 9 AM - 6 PM\n\n"
            "They can review your case and work toward a resolution. "
            "Is there anything else I can help you with regarding our "
            "vehicles or services?"
        )

    # Guardrail 2: Off-topic detection
    off_topic_keywords = [
        "recipe", "weather", "stock market", "sports score",
        "homework", "write me a poem", "tell me a joke",
        "who is the president", "capital of",
    ]
    if any(kw in query_lower for kw in off_topic_keywords):
        return (
            "I appreciate the question, but I am specifically designed to "
            "help with AutoElite Motors -- our vehicles, financing, "
            "promotions, and services. Is there anything car-related I "
            "can help you with today?"
        )

    return None


# -------------------------------------------------------------------------
# Context Assembly
# -------------------------------------------------------------------------

def build_context(system_prompt: str, history: list, retrieved_docs: list,
                  user_query: str) -> tuple:
    """
    Assemble the full LLM context from system prompt, history, retrieved
    documents, and the current user query.

    This is the core of context engineering: carefully structuring what
    goes into the LLM's context window. The order matters -- documents
    come first so the LLM sees the evidence before the question, and
    history provides conversation continuity.

    Returns:
        A tuple of (messages_list, token_breakdown_dict).
    """
    # Build document context string
    if retrieved_docs:
        docs_text = "Here is relevant information from our records:\n\n"
        for i, doc in enumerate(retrieved_docs, 1):
            docs_text += (
                f"[Document {i}] (relevance: {doc['distance']:.3f})\n"
                f"{doc['text']}\n\n"
            )
    else:
        docs_text = "No specific documents were found for this query."

    # Build history text from the conversation window
    history_text = ""
    if history:
        history_text = "Previous conversation:\n"
        for msg in history:
            role = "Customer" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    # Combine into a single user message with structured sections
    full_user_content = ""
    if docs_text:
        full_user_content += docs_text + "\n"
    if history_text:
        full_user_content += history_text + "\n"
    full_user_content += f"Current question: {user_query}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=full_user_content),
    ]

    # Calculate token breakdown for the monitoring footer
    system_tokens = count_tokens(system_prompt)
    docs_tokens = count_tokens(docs_text) if retrieved_docs else 0
    history_tokens = count_tokens(history_text) if history else 0
    query_tokens = count_tokens(user_query)
    total_tokens = system_tokens + docs_tokens + history_tokens + query_tokens

    breakdown = {
        "system_prompt": system_tokens,
        "retrieved_docs": docs_tokens,
        "chat_history": history_tokens,
        "user_query": query_tokens,
        "total": total_tokens,
    }

    return messages, breakdown


# -------------------------------------------------------------------------
# Chainlit Event Handlers
# -------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session.

    This runs once when a user first connects. It:
      1. Sets up the ChromaDB collections (loads all documents)
      2. Initializes empty conversation history
      3. Creates the settings UI (mode and model selection)
      4. Sends a welcome message with Phoenix status
    """
    # Set up ChromaDB collections
    customer_col, employee_col = setup_collections()
    cl.user_session.set("customer_collection", customer_col)
    cl.user_session.set("employee_collection", employee_col)

    # Initialize conversation history as an empty list
    cl.user_session.set("history", [])

    # Default settings
    cl.user_session.set("mode", "customer")
    cl.user_session.set("model", DEFAULT_MODEL)

    # Create the settings UI with mode and model dropdowns
    settings = cl.ChatSettings([
        cl.input_widget.Select(
            id="mode",
            label="Mode",
            values=["customer", "employee"],
            initial_value="customer",
        ),
        cl.input_widget.Select(
            id="model",
            label="Model",
            values=list(MODELS.keys()),
            initial_value=list(MODELS.keys())[0],
        ),
    ])
    await settings.send()

    # Welcome message with Phoenix status
    mode = cl.user_session.get("mode")
    phoenix_note = ""
    if phoenix_session:
        phoenix_note = (
            f"\n\n_Monitoring active. "
            f"[Phoenix Dashboard]({PHOENIX_URL})_"
        )

    await cl.Message(
        content=(
            f"Welcome to AutoElite Motors! I am your **{mode}** assistant. "
            f"How can I help you today?\n\n"
            f"_Use the gear icon to switch between Customer and Employee "
            f"modes or change the AI model._"
            f"{phoenix_note}"
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """
    Handle mode and model changes from the settings UI.

    When the user switches mode (customer <-> employee), we clear the
    conversation history because the context and available data change.
    Model changes take effect on the next message without clearing history.
    """
    old_mode = cl.user_session.get("mode")
    new_mode = settings.get("mode", old_mode)
    cl.user_session.set("mode", new_mode)

    model_name = settings.get("model", list(MODELS.keys())[0])
    cl.user_session.set("model", MODELS.get(model_name, DEFAULT_MODEL))

    if new_mode != old_mode:
        # Clear history on mode switch -- the two modes have different
        # system prompts and data access, so old history would be confusing.
        cl.user_session.set("history", [])
        await cl.Message(
            content=(
                f"Switched to **{new_mode}** mode. "
                f"Conversation history has been cleared."
            )
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Process a user message through the full monitored RAG pipeline.

    Pipeline steps:
      1. Check guardrails (complaint redirect, off-topic detection)
      2. Retrieve relevant documents from ChromaDB
      3. Build the LLM context (system prompt + docs + history + query)
      4. Stream the LLM response
      5. Build and display the footer (sources, tokens, latency, Phoenix link)
      6. Update conversation history (window memory)

    Every step's timing is tracked for the latency display. If Phoenix is
    active, the retrieval and LLM calls are automatically traced via the
    OpenInference LangChain instrumentor.
    """
    request_start = time.time()

    query = message.content
    mode = cl.user_session.get("mode", "customer")
    model = cl.user_session.get("model", DEFAULT_MODEL)
    history = cl.user_session.get("history", [])

    # ----- Step 1: Guardrails -----
    guardrail_response = check_guardrails(query, mode)
    if guardrail_response:
        await cl.Message(content=guardrail_response).send()
        # Still track in history so the model has conversation context
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": guardrail_response})
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
        cl.user_session.set("history", history)
        return

    # ----- Step 2: Retrieve documents -----
    retrieval_start = time.time()

    if mode == "customer":
        collection = cl.user_session.get("customer_collection")
        system_prompt = CUSTOMER_PROMPT
    else:
        collection = cl.user_session.get("employee_collection")
        system_prompt = EMPLOYEE_PROMPT

    retrieved_docs = retrieve_relevant_docs(collection, query)
    retrieval_time = time.time() - retrieval_start

    # ----- Step 3: Build context -----
    messages, token_breakdown = build_context(
        system_prompt, history, retrieved_docs, query
    )

    # ----- Step 4: Stream the LLM response -----
    generation_start = time.time()

    llm = ChatTogether(
        model=model,
        temperature=0.3,
        max_tokens=512,
        streaming=True,
    )

    response_msg = cl.Message(content="")
    full_response = ""

    async for chunk in llm.astream(messages):
        token = chunk.content
        if token:
            full_response += token
            await response_msg.stream_token(token)

    generation_time = time.time() - generation_start
    total_latency = time.time() - request_start

    # ----- Step 5: Build the response footer -----
    # The footer provides at-a-glance monitoring information for every response:
    # which sources were used, how many tokens were consumed, how long it took,
    # and a link to the Phoenix dashboard for the full trace.
    response_tokens = count_tokens(full_response)
    token_breakdown["response"] = response_tokens
    token_breakdown["total"] += response_tokens

    # Collect unique source filenames from retrieved documents
    if retrieved_docs:
        seen_sources = []
        for doc in retrieved_docs:
            src = doc.get("source", "unknown")
            if src not in seen_sources:
                seen_sources.append(src)
        sources_str = ", ".join(seen_sources)
    else:
        sources_str = "none"

    # Assemble the compact footer line:
    # Sources: file1, file2 | Tokens: 1,234 | Latency: 1.2s | Phoenix Dashboard
    footer_line = (
        f"\n\n---\n"
        f"*Sources: {sources_str} "
        f"| Tokens: {token_breakdown['total']:,} "
        f"| Latency: {total_latency:.1f}s"
    )

    if phoenix_session:
        footer_line += f" | [Phoenix Dashboard]({PHOENIX_URL})"

    footer_line += "*"

    # Update the message with the footer appended
    response_msg.content = full_response + footer_line
    await response_msg.update()

    # ----- Step 6: Update conversation history -----
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": full_response})

    # Trim to window size (keep the most recent messages)
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    cl.user_session.set("history", history)
