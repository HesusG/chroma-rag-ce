# Phase 4: Chatbot UI

In this phase you move from command-line scripts to a full web-based chat interface.
You will learn how to load PDF documents, organize knowledge into separate ChromaDB
collections, and build a streaming chatbot UI with Chainlit.

By the end of this phase you will:

- Load and chunk PDF documents using LangChain's PyPDFLoader
- Organize documents into separate ChromaDB collections (external vs internal)
- Understand what Chainlit is and how it turns a Python script into a web app
- Build a chat interface with streaming responses and a Customer/Employee mode toggle
- Run a fully working RAG chatbot in your browser

---

## Table of Contents

1. [Loading PDF Documents with LangChain](#loading-pdf-documents-with-langchain)
2. [Organizing Knowledge into Collections](#organizing-knowledge-into-collections)
3. [Introduction to Chainlit](#introduction-to-chainlit)
4. [Building the Chat Interface](#building-the-chat-interface)
5. [Customer and Employee Modes](#customer-and-employee-modes)
6. [Setup](#setup)
7. [Running the Scripts](#running-the-scripts)

---

## Loading PDF Documents with LangChain

So far in this project you have loaded text files, CSV files, JSON, and Markdown.
Real-world knowledge bases almost always include PDFs -- training manuals, contracts,
product specs, compliance documents, and so on. PDFs are harder to work with because
they are not plain text; they contain layout information, fonts, images, and sometimes
scanned content.

LangChain provides `PyPDFLoader` from `langchain_community.document_loaders` to handle
this. Under the hood it uses the `pypdf` library to extract text page by page.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/training_manual.pdf")
pages = loader.load()

# Each page becomes a Document with:
#   - page_content: the extracted text
#   - metadata: {"source": "data/training_manual.pdf", "page": 0}
```

Key things to know about PDF loading:

- **One Document per page.** PyPDFLoader creates one LangChain Document object for
  each page of the PDF. A 20-page PDF produces 20 Documents.
- **Text extraction is imperfect.** PDFs store layout, not semantic structure. Tables
  may come out garbled. Scanned PDFs (images of text) will return empty strings unless
  you use OCR.
- **Chunking is still needed.** A single page can contain 500+ words. You should still
  run the extracted text through a text splitter before adding it to your vector
  database.
- **Metadata is automatic.** Each Document includes the source file path and page
  number, which is useful for citations.

Script `01_pdf_loading.py` demonstrates this workflow end to end.

---

## Organizing Knowledge into Collections

In earlier phases, all documents went into a single ChromaDB collection. That works
for simple use cases, but real applications often need to separate knowledge by access
level, department, or topic.

In this phase we introduce **two collections**:

| Collection | Purpose | Contents |
|---|---|---|
| `external_knowledge` | Customer-facing information | Inventory, policies, promotions, FAQ |
| `internal_knowledge` | Employee-only information | Customer complaints, training manual |

This separation matters for several reasons:

- **Access control.** Customers should not see internal complaint data or training
  procedures. By putting internal documents in a separate collection, you can control
  which collection gets searched based on who is asking.
- **Relevance.** Searching a smaller, focused collection is more likely to return
  relevant results than searching everything at once.
- **Maintainability.** You can update internal and external knowledge independently.

Script `02_collections.py` shows how to create and query these collections.

---

## Introduction to Chainlit

Up to this point, every script in this project has been a command-line program. You
type a question, press Enter, and see the response in your terminal. That works for
learning, but it is not how you would deploy a chatbot for real users.

**Chainlit** is an open-source Python framework that turns a Python script into a
web-based chat interface. It handles the hard parts of building a chat UI:

- A clean message-based interface (similar to ChatGPT)
- Streaming tokens as the LLM generates them (no waiting for the full response)
- Session management (each browser tab gets its own conversation)
- Settings panels for user configuration (model selection, mode toggles)
- Step visualization (showing retrieval and processing steps)

### How Chainlit Works

Chainlit uses **decorator functions** to define your chatbot's behavior:

```python
import chainlit as cl

@cl.on_chat_start
async def start():
    """Runs once when a user opens the chat."""
    # Initialize collections, set up session state
    await cl.Message(content="Welcome!").send()

@cl.on_message
async def main(message: cl.Message):
    """Runs every time the user sends a message."""
    # Retrieve documents, call LLM, stream response
    msg = cl.Message(content="")
    await msg.stream_token("Hello ")
    await msg.stream_token("world!")
    await msg.send()

@cl.on_settings_update
async def settings_update(settings):
    """Runs when the user changes settings in the UI."""
    # Update mode, model, etc.
```

Key concepts:

- **`@cl.on_chat_start`** -- called once when the user opens the chat. Use it to load
  data, set up collections, and send a welcome message.
- **`@cl.on_message`** -- called every time the user sends a message. This is where
  your RAG pipeline lives: retrieve, build context, call the LLM, stream the response.
- **`@cl.on_settings_update`** -- called when the user changes a setting in the UI
  sidebar (like switching from Customer to Employee mode).
- **`cl.user_session`** -- per-user session storage. Each browser tab gets its own
  session. Use it to store the selected mode, model, and chat history.
- **`cl.Message`** -- the message object. Supports streaming via `stream_token()`.
- **`cl.Step`** -- shows a collapsible step in the UI (useful for showing retrieval).
- **`cl.ChatSettings`** -- defines a settings panel with dropdowns, toggles, etc.

### From CLI to Web UI

Running a Chainlit app is one command:

```bash
chainlit run 03_chainlit_app.py
```

This starts a local web server (default: http://localhost:8000) and opens your browser.
No HTML, CSS, or JavaScript required -- Chainlit generates the entire UI from your
Python decorators.

---

## Building the Chat Interface

The main chatbot (`03_chainlit_app.py`) combines everything from this project so far:

1. **Document loading** -- loads all data files (CSV, JSON, TXT, MD, PDF) into memory
2. **ChromaDB collections** -- creates external and internal collections
3. **Retrieval** -- searches the appropriate collection based on the user's mode
4. **LLM generation** -- sends the retrieved context + question to the LLM
5. **Streaming** -- streams the response token by token to the UI
6. **Source attribution** -- shows which documents were used to generate the answer

The architecture looks like this:

```
User types a question in the browser
       |
       v
Chainlit receives the message
       |
       v
Determine mode (Customer or Employee)
       |
       v
Query the appropriate ChromaDB collection(s)
       |
       v
Build a prompt with system instructions + retrieved context + user question
       |
       v
Stream the LLM response back to the browser
       |
       v
Append source citations
```

---

## Customer and Employee Modes

The chatbot supports two modes, selectable via the settings panel (gear icon):

**Customer Mode:**
- Searches only the `external_knowledge` collection
- System prompt is friendly and customer-oriented
- Can answer questions about inventory, pricing, financing, promotions, and policies
- Does NOT have access to internal complaints or training materials

**Employee Mode:**
- Searches BOTH the `internal_knowledge` and `external_knowledge` collections
- System prompt is professional and employee-oriented
- Can answer questions about complaints, training procedures, inventory, and policies
- Has access to everything

This is a simple but effective form of access control. In a production system you
would use authentication and role-based access, but the collection-separation pattern
is the same.

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A [Together.ai](https://together.ai) API key (free tier available)
- All dependencies from earlier phases plus Chainlit and pypdf

### 1. Install dependencies

```bash
cd phase-4-chatbot-ui
pip install -r requirements.txt
```

### 2. Configure your environment

Create a `.env` file in the `phase-4-chatbot-ui/` directory:

```bash
TOGETHER_API_KEY=your-api-key-here
```

Or export it in your terminal:

```bash
export TOGETHER_API_KEY="your-api-key-here"
```

### 3. (Optional) Add a training manual PDF

The chatbot will look for `data/training_manual.pdf` as an internal document. If you
do not have one, the app will simply skip it. You can create a simple PDF using any
word processor, or download a sample training manual from the internet.

---

## Running the Scripts

### Script 1: PDF Loading (`01_pdf_loading.py`)

Demonstrates how to load and chunk a PDF document using LangChain's PyPDFLoader. If
the training manual PDF does not exist, the script will print a helpful message and
exit gracefully.

```bash
python 01_pdf_loading.py
```

No API key needed.

### Script 2: Collections (`02_collections.py`)

Shows how to organize documents into separate ChromaDB collections and query them
independently. Includes a CLI mode selector to simulate Customer vs Employee access.

```bash
python 02_collections.py
```

No API key needed.

### Script 3: Chainlit App (`03_chainlit_app.py`)

The full chatbot with web UI. This is the main deliverable of Phase 4.

```bash
chainlit run 03_chainlit_app.py
```

Requires `TOGETHER_API_KEY` to be set. Opens in your browser at http://localhost:8000.

Use the gear icon in the sidebar to:
- Switch between **Customer** and **Employee** mode
- Select your preferred AI model

Example questions to try:

**As a Customer:**
- "What SUVs do you have under $40,000?"
- "Tell me about your financing options"
- "What promotions are running right now?"
- "What is your return policy?"

**As an Employee:**
- "Show me recent customer complaints"
- "What complaints have we received about service delays?"
- "What is the return policy procedure?"
- "Are there any unresolved complaints?"

---

## What You Learned

After completing this phase, you understand:

- **PyPDFLoader** extracts text from PDFs page by page, with automatic metadata
- **Separate collections** let you control access and improve retrieval relevance
- **Chainlit** turns a Python script into a web chat interface with one command
- **Streaming** sends tokens to the user as they are generated, not all at once
- **Settings panels** let users configure the app without touching code
- **Mode toggling** is a simple form of access control via collection selection

Next up: [Phase 5 - Context Engineering](../phase-5-context-engineering/) where you
will learn about system prompts, memory management, and context window optimization.
