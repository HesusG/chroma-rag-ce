"""
Phase 5 - Script 04: Chainlit App with Context Engineering
============================================================
Enhanced Chainlit chatbot that applies context engineering techniques:
  - XML-tagged system prompts per mode (customer / employee)
  - Window-based conversation memory (last 10 messages)
  - Relevance-filtered retrieval from ChromaDB
  - Guardrails: off-topic redirection, complaint handling, mode-aware detail
  - Token usage monitoring displayed in response footer

Run:
    chainlit run 04_chainlit_app.py -w
"""

import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv

import chromadb
import chainlit as cl
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split())

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
MODELS = {
    "Llama 3.3 70B (Free)": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "DeepSeek R1 70B (Free)": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Llama 3.1 8B": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
}
MAX_HISTORY_MESSAGES = 10
RELEVANCE_THRESHOLD = 1.5  # ChromaDB L2 distance; lower = more relevant

# ---------------------------------------------------------------------------
# System Prompts (XML-tagged)
# ---------------------------------------------------------------------------

CUSTOMER_SYSTEM_PROMPT = """<role>
You are a friendly and knowledgeable sales assistant at AutoElite Motors, a trusted car dealership.
</role>

<task>
Help customers find the right vehicle, understand financing options, learn about current promotions, and answer questions about dealership policies. Use the provided context to give accurate, helpful responses.
</task>

<constraints>
- Only answer questions related to AutoElite Motors, cars, financing, and automotive topics
- If asked about unrelated topics, politely redirect to how you can help with their car needs
- Never fabricate inventory, prices, or terms not in the context
- If unsure, suggest the customer contact the dealership directly
- Do not share internal employee information with customers
</constraints>

<output_format>
- Lead with a direct, helpful answer
- Use bullet points for multiple options or features
- Include specific details (prices, years, features) when available
- End with a suggestion or next step (e.g., schedule a test drive)
- Keep responses concise but informative (2-4 paragraphs)
</output_format>"""

EMPLOYEE_SYSTEM_PROMPT = """<role>
You are an internal assistant for AutoElite Motors employees. You help staff access information efficiently.
</role>

<task>
Help employees find information about customer complaints, training procedures, inventory status, support tickets, and internal policies. Provide data-driven, actionable responses.
</task>

<constraints>
- Provide factual, concise answers based on the context
- When discussing complaints or tickets, include relevant details (dates, status, categories)
- Flag urgent or escalated items
- If information is not in the context, say so clearly
</constraints>

<output_format>
- Be concise and data-driven
- Use tables or structured lists for multiple items
- Include status, dates, and categories when relevant
- Highlight action items or items needing attention
</output_format>"""

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_inventory() -> list:
    """Load inventory data from CSV."""
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
            metas.append({"source": "inventory", "category": row.get("category", "")})
            ids.append(f"inv_{i}")
    return docs, metas, ids


def load_complaints() -> list:
    """Load complaint data from CSV."""
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
            metas.append({"source": "complaints", "status": row.get("status", "")})
            ids.append(f"comp_{i}")
    return docs, metas, ids


def load_policies() -> list:
    """Load policies from text file, split into sections."""
    docs, metas, ids = [], [], []
    path = DATA_DIR / "policies.txt"
    if not path.exists():
        return docs, metas, ids
    with open(path, "r") as f:
        content = f.read()
    # Split on section headers (lines of dashes)
    sections = content.split("\n\n")
    current_section = ""
    section_idx = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue
        current_section += section + "\n\n"
        # When we hit a section break, save and reset
        if len(current_section) > 200:
            docs.append(current_section.strip())
            metas.append({"source": "policies"})
            ids.append(f"pol_{section_idx}")
            section_idx += 1
            current_section = ""
    if current_section.strip():
        docs.append(current_section.strip())
        metas.append({"source": "policies"})
        ids.append(f"pol_{section_idx}")
    return docs, metas, ids


def load_promotions() -> list:
    """Load promotions from markdown file."""
    docs, metas, ids = [], [], []
    path = DATA_DIR / "promotions.md"
    if not path.exists():
        return docs, metas, ids
    with open(path, "r") as f:
        content = f.read()
    # Split on ## headers
    sections = content.split("## ")
    for i, section in enumerate(sections):
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        docs.append(section)
        metas.append({"source": "promotions"})
        ids.append(f"promo_{i}")
    return docs, metas, ids


def load_faq() -> list:
    """Load FAQ from JSON file."""
    docs, metas, ids = [], [], []
    path = DATA_DIR / "faq.json"
    if not path.exists():
        return docs, metas, ids
    with open(path, "r") as f:
        faqs = json.load(f)
    for i, faq in enumerate(faqs):
        text = f"Q: {faq['question']}\nA: {faq['answer']}"
        docs.append(text)
        metas.append({"source": "faq", "category": faq.get("category", "")})
        ids.append(f"faq_{i}")
    return docs, metas, ids


# ---------------------------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------------------------

def setup_collections():
    """Create and populate ChromaDB collections."""
    client = chromadb.Client()

    # Customer-facing collection
    customer_col = client.get_or_create_collection("customer_docs")
    if customer_col.count() == 0:
        all_docs, all_metas, all_ids = [], [], []
        for loader in [load_inventory, load_policies, load_promotions, load_faq]:
            docs, metas, ids = loader()
            all_docs.extend(docs)
            all_metas.extend(metas)
            all_ids.extend(ids)
        if all_docs:
            customer_col.add(documents=all_docs, metadatas=all_metas, ids=all_ids)

    # Employee collection (includes complaints)
    employee_col = client.get_or_create_collection("employee_docs")
    if employee_col.count() == 0:
        all_docs, all_metas, all_ids = [], [], []
        for loader in [load_inventory, load_complaints, load_policies, load_promotions, load_faq]:
            docs, metas, ids = loader()
            all_docs.extend(docs)
            all_metas.extend(metas)
            all_ids.extend(ids)
        if all_docs:
            employee_col.add(documents=all_docs, metadatas=all_metas, ids=all_ids)

    return customer_col, employee_col


# ---------------------------------------------------------------------------
# Retrieval with relevance filtering
# ---------------------------------------------------------------------------

def retrieve_relevant_docs(collection, query: str, n_results: int = 8) -> list:
    """Retrieve documents and filter by relevance threshold."""
    results = collection.query(query_texts=[query], n_results=n_results)

    filtered_docs = []
    if results and results["documents"] and results["distances"]:
        for doc, distance in zip(results["documents"][0], results["distances"][0]):
            if distance <= RELEVANCE_THRESHOLD:
                filtered_docs.append({"text": doc, "distance": distance})

    # Sort by relevance (lowest distance first)
    filtered_docs.sort(key=lambda x: x["distance"])
    return filtered_docs


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def check_guardrails(query: str, mode: str) -> str | None:
    """Check if the query needs special handling. Returns a redirect message
    or None if the query should be processed normally."""

    query_lower = query.lower()

    # Check for complaint-related keywords in customer mode
    complaint_keywords = ["complaint", "complain", "unhappy", "frustrated",
                          "terrible", "worst", "sue", "lawyer", "refund",
                          "scam", "ripped off", "angry"]
    if mode == "customer" and any(kw in query_lower for kw in complaint_keywords):
        return (
            "I understand you have a concern, and I am sorry to hear that. "
            "For complaints and escalations, I recommend contacting our "
            "Customer Support team directly:\n\n"
            "- **Phone:** (555) 123-4567\n"
            "- **Email:** support@autoelitemotors.com\n"
            "- **Hours:** Monday-Friday, 9 AM - 6 PM\n\n"
            "They can review your case and work toward a resolution. "
            "Is there anything else I can help you with regarding our vehicles or services?"
        )

    # Check for clearly off-topic queries
    off_topic_keywords = ["recipe", "weather", "stock market", "sports score",
                          "homework", "write me a poem", "tell me a joke",
                          "who is the president", "capital of"]
    if any(kw in query_lower for kw in off_topic_keywords):
        return (
            "I appreciate the question, but I am specifically designed to help "
            "with AutoElite Motors -- our vehicles, financing, promotions, and "
            "services. Is there anything car-related I can help you with today?"
        )

    return None


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def build_context(system_prompt: str, history: list, retrieved_docs: list,
                  user_query: str) -> tuple:
    """Assemble the full context and return (messages, token_breakdown)."""

    # Build document context string
    if retrieved_docs:
        docs_text = "Here is relevant information from our records:\n\n"
        for i, doc in enumerate(retrieved_docs, 1):
            docs_text += f"[Document {i}] (relevance: {doc['distance']:.3f})\n{doc['text']}\n\n"
    else:
        docs_text = "No specific documents were found for this query."

    # Build history text for the system message
    history_text = ""
    if history:
        history_text = "Previous conversation:\n"
        for msg in history:
            role = "Customer" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    # Assemble the full user message with context
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

    # Calculate token breakdown
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


# ---------------------------------------------------------------------------
# Chainlit event handlers
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Initialize the session: set up collections, history, and settings."""

    # Set up ChromaDB
    customer_col, employee_col = setup_collections()
    cl.user_session.set("customer_collection", customer_col)
    cl.user_session.set("employee_collection", employee_col)

    # Initialize conversation history
    cl.user_session.set("history", [])

    # Default settings
    cl.user_session.set("mode", "customer")
    cl.user_session.set("model", DEFAULT_MODEL)

    # Create settings UI
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

    mode = cl.user_session.get("mode")
    await cl.Message(
        content=f"Welcome to AutoElite Motors! I am your **{mode}** assistant. How can I help you today?\n\n_Use the gear icon to switch between Customer and Employee modes or change the AI model._"
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle mode and model changes."""
    old_mode = cl.user_session.get("mode")
    new_mode = settings.get("mode", old_mode)
    cl.user_session.set("mode", new_mode)

    model_name = settings.get("model", list(MODELS.keys())[0])
    cl.user_session.set("model", MODELS.get(model_name, DEFAULT_MODEL))

    if new_mode != old_mode:
        # Clear history on mode switch
        cl.user_session.set("history", [])
        await cl.Message(
            content=f"Switched to **{new_mode}** mode. Conversation history has been cleared."
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Process user message with full context engineering pipeline."""

    query = message.content
    mode = cl.user_session.get("mode", "customer")
    model = cl.user_session.get("model", DEFAULT_MODEL)
    history = cl.user_session.get("history", [])

    # Step 1: Check guardrails
    guardrail_response = check_guardrails(query, mode)
    if guardrail_response:
        await cl.Message(content=guardrail_response).send()
        # Still add to history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": guardrail_response})
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
        cl.user_session.set("history", history)
        return

    # Step 2: Retrieve relevant documents
    if mode == "customer":
        collection = cl.user_session.get("customer_collection")
        system_prompt = CUSTOMER_SYSTEM_PROMPT
    else:
        collection = cl.user_session.get("employee_collection")
        system_prompt = EMPLOYEE_SYSTEM_PROMPT

    retrieved_docs = retrieve_relevant_docs(collection, query)

    # Step 3: Build context
    messages, token_breakdown = build_context(
        system_prompt, history, retrieved_docs, query
    )

    # Step 4: Stream the response
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

    # Step 5: Build footer with sources and token info
    response_tokens = count_tokens(full_response)
    token_breakdown["response"] = response_tokens
    token_breakdown["total"] += response_tokens

    footer_parts = []

    # Source documents
    if retrieved_docs:
        footer_parts.append("\n\n---\n**Sources:**")
        seen_sources = set()
        for doc in retrieved_docs[:5]:
            source_preview = doc["text"][:80].replace("\n", " ")
            if source_preview not in seen_sources:
                seen_sources.add(source_preview)
                footer_parts.append(f"- {source_preview}... _(relevance: {doc['distance']:.3f})_")

    # Token usage
    footer_parts.append(f"\n**Token Usage:**")
    footer_parts.append(
        f"System: {token_breakdown['system_prompt']} | "
        f"Docs: {token_breakdown['retrieved_docs']} | "
        f"History: {token_breakdown['chat_history']} | "
        f"Query: {token_breakdown['user_query']} | "
        f"Response: {token_breakdown['response']} | "
        f"**Total: {token_breakdown['total']}**"
    )

    response_msg.content = full_response + "\n".join(footer_parts)
    await response_msg.update()

    # Step 6: Update conversation history (window memory)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": full_response})

    # Trim to window size
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    cl.user_session.set("history", history)
