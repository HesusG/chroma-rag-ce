# Phase 1: Foundations

Welcome to Phase 1 of the ChromaDB RAG Chatbot project. In this phase you will learn
the core concepts behind RAG (Retrieval-Augmented Generation) and build your first
working RAG pipeline from scratch.

By the end of this phase you will:

- Understand what RAG is and why it matters
- Store and query documents in a vector database (ChromaDB)
- Build an interactive chatbot powered by retrieved context and an LLM

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [What are Embeddings?](#what-are-embeddings)
3. [What is a Vector Database / ChromaDB?](#what-is-a-vector-database--chromadb)
4. [What is Context Engineering?](#what-is-context-engineering)
5. [The 6 Components of AI Agents](#the-6-components-of-ai-agents)
6. [Setup](#setup)
7. [Running the Scripts](#running-the-scripts)

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. It is a technique that makes LLMs
(Large Language Models) more useful by giving them access to your own data at query
time.

The problem RAG solves: LLMs are trained on public internet data. They do not know
about your company's inventory, policies, or internal documents. If you ask an LLM
"What SUVs do you have in stock?", it will hallucinate an answer because it has no
idea what your dealership carries.

RAG fixes this by **retrieving** relevant documents first, then **generating** an
answer grounded in those documents.

```
How RAG Works
=============

  User Question
       |
       v
  +-----------+      +-----------------+
  |  Retrieve  | ---> | Vector Database |
  |  (search)  | <--- | (your documents)|
  +-----------+      +-----------------+
       |
       | relevant documents
       v
  +-----------+
  |  Augment   |  Combine the question + retrieved documents
  |  (format)  |  into a prompt for the LLM
  +-----------+
       |
       v
  +-----------+
  | Generate  |  LLM reads the context and writes
  | (LLM)     |  a grounded answer
  +-----------+
       |
       v
    Answer
```

Without RAG, the LLM guesses. With RAG, the LLM reads your actual data and answers
based on facts.

---

## What are Embeddings?

Embeddings are the key technology that makes RAG possible. An embedding is a way of
**converting text into a list of numbers (a vector) that captures its meaning**.

Think of it this way:

- The sentence "affordable family SUV" might become `[0.12, -0.87, 0.45, ...]`
- The sentence "budget-friendly crossover for families" would get a very similar
  set of numbers, because the *meaning* is similar
- The sentence "Italian pasta recipe" would get a very different set of numbers

When two pieces of text have similar meaning, their embedding vectors are close
together in space. When meanings are different, the vectors are far apart.

```
  Meaning Space (simplified to 2D)
  ================================

    "family SUV"  *----*  "crossover for families"
                    (close together = similar meaning)



                          * "pasta recipe"
                    (far away = different meaning)
```

ChromaDB generates embeddings automatically using a built-in model, so you do not
need to worry about this step yet. You just add text, and ChromaDB handles the rest.

---

## What is a Vector Database / ChromaDB?

A **vector database** stores documents along with their embedding vectors. When you
search, it does not look for exact keyword matches (like a traditional database).
Instead, it finds documents whose *meaning* is closest to your query.

This is called **semantic search** -- searching by meaning rather than by keywords.

**ChromaDB** is an open-source vector database that is easy to get started with:

- Runs in-memory (no server setup needed)
- Generates embeddings automatically
- Supports metadata filtering (e.g., filter by price range, year, make)
- Works well with Python and LangChain

In this phase, we use ChromaDB's ephemeral (in-memory) mode. Your data lives only
for the duration of the script. Later phases will use persistent storage.

---

## What is Context Engineering?

Context engineering is the practice of **getting the right information, in the right
format, at the right time into the LLM's context window**.

Every LLM has a context window -- a limit on how much text it can read at once. If
you dump too much irrelevant text in, the LLM gets confused. If you leave out
critical information, the LLM hallucinates.

Context engineering is about being deliberate:

- **Right information** -- retrieve only the documents relevant to the question
- **Right format** -- structure the context so the LLM can easily parse it
- **Right time** -- provide information when the LLM needs it, not all upfront

RAG is one of the most important tools in context engineering. In Phase 5, we will
explore this topic in much greater depth, including system prompts, memory management,
and context window optimization.

---

## The 6 Components of AI Agents

As you work through this project, it helps to know the bigger picture. AI agents are
built from 6 core components:

| Component       | What It Does                                              |
|-----------------|-----------------------------------------------------------|
| **Model**       | The LLM that reasons and generates text (e.g., Llama 3)  |
| **Tools**       | Functions the agent can call (e.g., search, calculate)    |
| **Knowledge**   | Data the agent can retrieve (this is where RAG lives)     |
| **Audio**       | Voice input/output capabilities                           |
| **Guardrails**  | Safety rules and output validation                        |
| **Orchestration**| Logic that coordinates the agent's workflow               |

In this phase we focus on **Knowledge** (ChromaDB + document retrieval) and **Model**
(Together.ai LLM). Later phases add tools, guardrails, and orchestration.

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A [Together.ai](https://together.ai) account (free tier available)

### 1. Install dependencies

```bash
cd phase-1-foundations
pip install -r requirements.txt
```

### 2. Get your Together.ai API key

1. Go to [https://api.together.ai/](https://api.together.ai/) and create a free account
2. Navigate to **API Keys** in your dashboard
3. Create a new key and copy it

### 3. Configure your environment

Create a `.env` file in the `phase-1-foundations/` directory:

```bash
TOGETHER_API_KEY=your-api-key-here
```

Alternatively, export it in your terminal:

```bash
export TOGETHER_API_KEY="your-api-key-here"
```

The `.env` file is already in `.gitignore` so your key will not be committed to Git.

---

## Running the Scripts

### Script 1: Hello ChromaDB (`01_hello_chroma.py`)

This script introduces ChromaDB without any LLM involvement. It:

- Creates an in-memory ChromaDB database
- Loads the 10 cars from `data/inventory.txt`
- Adds them to a vector collection with metadata (make, model, year, price)
- Runs two semantic search queries and prints the results

```bash
python 01_hello_chroma.py
```

No API key needed for this script. It uses ChromaDB's built-in embedding model.

### Script 2: Basic RAG (`02_basic_rag.py`)

This script builds a complete RAG pipeline. It:

- Loads both inventory and policy documents into ChromaDB
- Connects to Together.ai to use an LLM (default: Llama 3.3 70B)
- Runs an interactive chat loop where you ask questions and get RAG-powered answers
- Shows you which documents were retrieved so you can see the RAG pipeline in action

```bash
python 02_basic_rag.py
```

Requires `TOGETHER_API_KEY` to be set (via `.env` file or environment variable).

Type your questions at the prompt. Type `quit` or `exit` to stop.

Example questions to try:

- "What SUVs do you have?"
- "Tell me about your financing options"
- "Do you have anything under $30,000?"
- "What is your return policy?"
- "What certified pre-owned vehicles are available?"

---

## What You Learned

After completing this phase, you understand:

- **RAG** is retrieve-then-generate: find relevant documents, then ask the LLM
- **Embeddings** turn text into numbers that capture meaning
- **ChromaDB** stores documents and finds similar ones via semantic search
- **Context engineering** is about feeding the LLM the right information
- **Together.ai** provides free LLM access for building RAG applications

Next up: [Phase 2 - Document Loading](../phase-2-document-loading/) where you will
load multiple file formats (CSV, JSON, Markdown) and learn about chunking strategies.
