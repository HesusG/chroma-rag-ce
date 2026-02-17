"""
Phase 4 - Script 3: Chainlit Chatbot App
==========================================

This is the main deliverable of Phase 4 -- a full web-based RAG chatbot built
with Chainlit. It combines everything from the project so far:

  1. Loads documents from multiple formats (CSV, JSON, TXT, MD, PDF)
  2. Organizes them into external (customer) and internal (employee) collections
  3. Retrieves relevant documents based on the user's question and mode
  4. Streams LLM responses in real time through a clean web interface
  5. Shows source attributions so users know where answers come from

Run with:
    chainlit run 03_chainlit_app.py

Then open http://localhost:8000 in your browser.

Use the gear icon to switch between Customer and Employee mode, and to
select your preferred AI model.
"""

import os
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
import chromadb
import chainlit as cl
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file (for TOGETHER_API_KEY)
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# Available models for the UI dropdown. Keys are display names, values are
# Together.ai model IDs. The free models have no usage cost; the others
# use your free credit balance.
MODELS = {
    "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "DeepSeek R1 70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Llama 3.1 8B": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
}


# ---------------------------------------------------------------------------
# Document Loading Helpers
# ---------------------------------------------------------------------------
# Each function reads a specific file format and returns a list of
# (content, metadata) tuples. This keeps the loading logic simple and
# format-specific, while the collection setup code stays clean.

def load_text_file(filepath):
    """Load a plain text file and return as a single document."""
    with open(filepath, "r") as f:
        content = f.read()
    return [(content, {"source": filepath.name, "type": "text"})]


def load_csv_file(filepath):
    """Load a CSV file where each row becomes a separate document."""
    docs = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = ", ".join(f"{k}: {v}" for k, v in row.items())
            metadata = {"source": filepath.name, "type": "csv"}
            # Preserve useful fields as metadata for potential filtering
            for key in ["make", "model", "year", "price", "category", "status"]:
                if key in row:
                    metadata[key] = row[key]
            docs.append((content, metadata))
    return docs


def load_json_file(filepath):
    """Load a JSON FAQ file where each Q&A pair becomes a document."""
    with open(filepath, "r") as f:
        data = json.load(f)
    docs = []
    for item in data:
        content = f"Question: {item['question']}\nAnswer: {item['answer']}"
        docs.append((content, {"source": filepath.name, "type": "json"}))
    return docs


def load_markdown_file(filepath):
    """Load a Markdown file and return as a single document."""
    with open(filepath, "r") as f:
        content = f.read()
    return [(content, {"source": filepath.name, "type": "markdown"})]


# ---------------------------------------------------------------------------
# ChromaDB Setup
# ---------------------------------------------------------------------------

def setup_collections():
    """
    Create and populate the external and internal ChromaDB collections.

    External collection (customer-facing):
        - inventory.csv -- vehicle listings
        - policies.txt -- dealership policies
        - promotions.md -- current promotions and offers
        - faq.json -- frequently asked questions

    Internal collection (employee-only):
        - complaints.csv -- customer complaints
        - training_manual.pdf -- employee training manual (if available)

    Returns:
        Tuple of (external_collection, internal_collection)
    """
    client = chromadb.Client()

    # --- Load external documents ---
    ext_docs = []
    ext_docs.extend(load_csv_file(DATA_DIR / "inventory.csv"))
    ext_docs.extend(load_text_file(DATA_DIR / "policies.txt"))
    ext_docs.extend(load_markdown_file(DATA_DIR / "promotions.md"))
    ext_docs.extend(load_json_file(DATA_DIR / "faq.json"))

    # Chunk large documents so embeddings are more precise
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunked_ext = []
    for content, metadata in ext_docs:
        if len(content) > 500:
            chunks = splitter.split_text(content)
            for chunk in chunks:
                chunked_ext.append((chunk, {**metadata, "access_level": "external"}))
        else:
            chunked_ext.append((content, {**metadata, "access_level": "external"}))

    # Create and populate the external collection
    external = client.get_or_create_collection("external_knowledge")
    external.add(
        documents=[d[0] for d in chunked_ext],
        metadatas=[d[1] for d in chunked_ext],
        ids=[f"ext_{i}" for i in range(len(chunked_ext))],
    )

    # --- Load internal documents ---
    int_docs = []

    # Load complaints CSV
    complaints_path = DATA_DIR / "complaints.csv"
    if complaints_path.exists():
        int_docs.extend(load_csv_file(complaints_path))

    # Load training manual PDF (optional -- skip gracefully if missing)
    training_path = DATA_DIR / "training_manual.pdf"
    if training_path.exists():
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(training_path))
            pages = loader.load()
            for page in pages:
                int_docs.append((
                    page.page_content,
                    {
                        "source": "training_manual.pdf",
                        "type": "pdf",
                        "page": str(page.metadata.get("page", 0)),
                    },
                ))
        except Exception:
            # PDF loading failed -- continue without it
            pass

    # Chunk internal documents
    chunked_int = []
    for content, metadata in int_docs:
        if len(content) > 500:
            chunks = splitter.split_text(content)
            for chunk in chunks:
                chunked_int.append((chunk, {**metadata, "access_level": "internal"}))
        else:
            chunked_int.append((content, {**metadata, "access_level": "internal"}))

    # Create and populate the internal collection
    internal = client.get_or_create_collection("internal_knowledge")
    if chunked_int:
        internal.add(
            documents=[d[0] for d in chunked_int],
            metadatas=[d[1] for d in chunked_int],
            ids=[f"int_{i}" for i in range(len(chunked_int))],
        )

    return external, internal


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
# These hold the ChromaDB collections after initialization. They are set once
# in on_chat_start and read on every message.

external_collection = None
internal_collection = None


# ---------------------------------------------------------------------------
# Chainlit Event Handlers
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def start():
    """
    Called once when a user opens the chat. This is where we:
    1. Load all documents into ChromaDB collections
    2. Set up the settings panel (mode + model selection)
    3. Initialize session state
    4. Send a welcome message
    """
    global external_collection, internal_collection

    # Show a loading message while we set up the knowledge base
    msg = cl.Message(content="Setting up knowledge base...")
    await msg.send()

    # Create and populate both collections
    external_collection, internal_collection = setup_collections()

    # Define the settings panel that appears when the user clicks the gear icon.
    # This gives users a way to switch modes and models without restarting.
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="user_mode",
                label="User Mode",
                values=["Customer", "Employee"],
                initial_value="Customer",
            ),
            cl.input_widget.Select(
                id="model",
                label="AI Model",
                values=list(MODELS.keys()),
                initial_value="Llama 3.3 70B",
            ),
        ]
    ).send()

    # Initialize per-session state. Each browser tab gets its own session,
    # so different users can have different modes and models.
    cl.user_session.set("user_mode", "Customer")
    cl.user_session.set("model", DEFAULT_MODEL)
    cl.user_session.set("chat_history", [])

    # Send the welcome message
    await cl.Message(
        content=(
            "Welcome to **AutoElite Motors** assistant! I'm ready to help.\n\n"
            "Use the settings panel (gear icon) to switch between **Customer** and "
            "**Employee** mode, and to select your preferred AI model.\n\n"
            "**Customer mode**: Ask about inventory, pricing, financing, promotions, "
            "and policies.\n"
            "**Employee mode**: Access complaints, training materials, and internal "
            "information."
        )
    ).send()


@cl.on_settings_update
async def settings_update(settings):
    """
    Called when the user changes settings in the UI sidebar.
    We update the session state so the next message uses the new settings.
    """
    cl.user_session.set("user_mode", settings["user_mode"])
    model_name = settings["model"]
    cl.user_session.set("model", MODELS.get(model_name, DEFAULT_MODEL))
    await cl.Message(
        content=f"Settings updated: **{settings['user_mode']}** mode, **{model_name}** model"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Called every time the user sends a message. This is the core RAG pipeline:
    1. Determine the user's mode (Customer or Employee)
    2. Query the appropriate ChromaDB collection(s)
    3. Build a prompt with system instructions + retrieved context
    4. Stream the LLM response back to the user
    5. Append source citations
    """
    user_mode = cl.user_session.get("user_mode", "Customer")
    model_id = cl.user_session.get("model", DEFAULT_MODEL)

    # Select which collection(s) to search based on mode.
    # - Customer: external only (no access to internal data)
    # - Employee: internal first, then also external (full access)
    if user_mode == "Employee":
        collection = internal_collection
        external = external_collection
    else:
        collection = external_collection
        external = None

    # ------------------------------------------------------------------
    # Step 1: Retrieve relevant documents
    # ------------------------------------------------------------------
    # cl.Step creates a collapsible section in the UI so the user can
    # see what the retrieval system found.
    async with cl.Step(name="Retrieving documents", type="retrieval") as step:
        results = collection.query(query_texts=[message.content], n_results=5)

        retrieved_docs = []
        if results and results["documents"]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                retrieved_docs.append({
                    "content": doc,
                    "source": metadata.get("source", "unknown"),
                    "distance": distance,
                })

        # For employees, also search the external collection to give them
        # access to customer-facing info (inventory, policies, etc.)
        if external is not None:
            ext_results = external.query(query_texts=[message.content], n_results=3)
            if ext_results and ext_results["documents"]:
                for doc, metadata, distance in zip(
                    ext_results["documents"][0],
                    ext_results["metadatas"][0],
                    ext_results["distances"][0],
                ):
                    retrieved_docs.append({
                        "content": doc,
                        "source": metadata.get("source", "unknown"),
                        "distance": distance,
                    })

        # Sort by distance (lower distance = more semantically similar)
        # and keep the top 5 most relevant documents.
        retrieved_docs.sort(key=lambda x: x["distance"])
        retrieved_docs = retrieved_docs[:5]

        step.output = f"Found {len(retrieved_docs)} relevant documents"

    # ------------------------------------------------------------------
    # Step 2: Format the context for the LLM
    # ------------------------------------------------------------------
    # Each retrieved document is labeled with its source so the LLM (and
    # the user) can trace where the information came from.
    context = "\n\n---\n\n".join(
        f"[Source: {doc['source']}]\n{doc['content']}"
        for doc in retrieved_docs
    )

    # ------------------------------------------------------------------
    # Step 3: Build the system prompt
    # ------------------------------------------------------------------
    # The system prompt sets the LLM's persona and instructions. It is
    # different for Customer and Employee modes.
    if user_mode == "Customer":
        system_prompt = (
            "You are a helpful assistant for AutoElite Motors, a car dealership. "
            "You help customers find cars, understand financing options, and learn "
            "about promotions and policies. Answer based on the provided context. "
            "If you don't know something, say so honestly. Be friendly and professional."
        )
    else:
        system_prompt = (
            "You are an internal assistant for AutoElite Motors employees. "
            "You help employees access information about customer complaints, "
            "training procedures, inventory, and policies. Answer based on the "
            "provided context. Be concise and professional."
        )

    # ------------------------------------------------------------------
    # Step 4: Call the LLM and stream the response
    # ------------------------------------------------------------------
    llm = ChatTogether(model=model_id)

    messages = [
        SystemMessage(content=f"{system_prompt}\n\nContext:\n{context}"),
        HumanMessage(content=message.content),
    ]

    # Stream tokens to the UI as they are generated. This gives the user
    # immediate feedback instead of waiting for the full response.
    msg = cl.Message(content="")
    async for chunk in llm.astream(messages):
        if chunk.content:
            await msg.stream_token(chunk.content)

    # ------------------------------------------------------------------
    # Step 5: Append source citations
    # ------------------------------------------------------------------
    # Show which data files contributed to the answer. This helps users
    # verify the information and builds trust in the system.
    sources = set(doc["source"] for doc in retrieved_docs)
    msg.content += f"\n\n---\n*Sources: {', '.join(sources)}*"
    await msg.update()
