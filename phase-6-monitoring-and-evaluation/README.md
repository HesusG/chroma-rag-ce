# Phase 6: Monitoring and Evaluation

In this phase you move from building a RAG chatbot to observing and evaluating it.
A RAG system that works in a demo is not the same as one that works in production.
Without monitoring, you cannot know if your retrieval is returning relevant documents,
if the LLM is hallucinating, how long each step takes, or when your knowledge base
has gone stale. This phase introduces the tools and techniques for answering those
questions.

By the end of this phase you will:

- Understand why monitoring matters for production RAG systems
- Set up Arize Phoenix for LLM observability and tracing
- Trace a full RAG pipeline: retrieval, context formatting, LLM generation
- Evaluate RAG quality with retrieval relevance and answer completeness metrics
- Build a fully monitored Chainlit chatbot with per-request telemetry

---

## Table of Contents

1. [Why Monitoring Matters](#why-monitoring-matters)
2. [Introduction to Arize Phoenix and OpenTelemetry](#introduction-to-arize-phoenix-and-opentelemetry)
3. [What to Trace](#what-to-trace)
4. [Evaluating RAG Quality](#evaluating-rag-quality)
5. [Data Freshness](#data-freshness)
6. [The RAG Pipeline as a Traceable System](#the-rag-pipeline-as-a-traceable-system)
7. [Setup](#setup)
8. [Running the Scripts](#running-the-scripts)

---

## Why Monitoring Matters

Consider what happens when you deploy a RAG chatbot without monitoring:

- A customer asks about a vehicle that was sold last week. The knowledge base still
  lists it as available. The chatbot confidently tells the customer to come in and
  test drive it. The customer drives 45 minutes to the dealership and finds out the
  car is gone. You have no idea this happened until they complain.

- Your embedding model returns documents about financing when the customer asked about
  electric vehicles. The LLM generates a plausible-sounding answer that mixes financing
  details with EV information. The customer gets confused. You never see this because
  the response looks grammatically correct.

- Response times creep from 2 seconds to 8 seconds over a month because the ChromaDB
  collection grew and nobody noticed. Customers start abandoning the chat.

Monitoring gives you visibility into what your system is actually doing, not what you
hope it is doing. It turns a black box into a glass box. Specifically, monitoring a
RAG system means tracking:

- **Retrieval quality**: Are the right documents being found for each query?
- **Generation quality**: Are the LLM responses accurate and grounded in the context?
- **Latency**: How long does each step take? Where are the bottlenecks?
- **Token usage**: How many tokens are being consumed per request? What does it cost?
- **Failure modes**: When does the system fail, and how does it fail?

---

## Introduction to Arize Phoenix and OpenTelemetry

**Arize Phoenix** is an open-source LLM observability tool. It provides a local
dashboard where you can inspect every call your LLM application makes: what went in,
what came out, how long it took, and how many tokens it used.

Phoenix is built on **OpenTelemetry (OTel)**, the industry-standard framework for
distributed tracing. OpenTelemetry defines a common format for "spans" -- units of
work in your application. Each span has a start time, end time, attributes (like
the model name or token count), and can be nested inside parent spans to show
the full call hierarchy.

Here is how the pieces fit together:

```
Your Python application
    |
    v
OpenInference Instrumentor (auto-instruments LangChain calls)
    |
    v
OpenTelemetry SDK (collects spans into traces)
    |
    v
Phoenix Collector (receives traces and stores them)
    |
    v
Phoenix Dashboard (web UI at localhost:6006)
```

The key insight is that you do not need to manually add tracing code to every
function. The `openinference-instrumentation-langchain` package automatically
instruments LangChain's LLM calls, embedding calls, retriever calls, and chain
executions. You set it up once at the start of your application, and every
LangChain operation is traced automatically.

### What Phoenix Shows You

When you open the Phoenix dashboard at `http://localhost:6006`, you see:

- **Traces**: Each user query becomes a trace. Click on a trace to see every span
  (retrieval, embedding, LLM call) with timing and input/output data.
- **Spans**: Individual operations within a trace. An LLM span shows the prompt,
  response, token counts, and latency. A retriever span shows the query and
  retrieved documents.
- **Projects**: Group traces by application or environment.
- **Evaluations**: Attach quality scores to traces (relevance, faithfulness, etc.).

---

## What to Trace

Not everything needs tracing, but in a RAG system these are the critical signals:

### Retrieval Latency
How long does it take to search your vector database? For ChromaDB with a few
hundred documents, this should be under 50ms. If it starts climbing, your collection
may be growing too large or your embedding model may be slow.

### LLM Latency
How long does the LLM take to generate a response? This depends on the model, the
prompt length, and the max tokens setting. Free-tier models can be slower during
peak hours. Track the P50 and P95 latencies to understand typical and worst-case
response times.

### Token Usage
Every LLM call consumes tokens. Track:
- **Input tokens**: How large is your prompt (system message + context + history)?
- **Output tokens**: How long are the generated responses?
- **Total tokens per request**: This determines your API cost.

If input tokens are growing over time, your context window may be filling up with
conversation history or overly large retrieved documents.

### Retrieval Relevance Scores
ChromaDB returns distance scores with each result. Lower distance means higher
similarity. Track whether the top-k results have low distances (good) or whether
the best match is still far from the query (bad). A consistently high distance for
the top result means your documents may not cover the topic the user is asking about.

---

## Evaluating RAG Quality

Monitoring tells you what happened. Evaluation tells you whether what happened was
good. There are three dimensions of RAG quality:

### Retrieval Relevance
Are the retrieved documents relevant to the question? This is the foundation -- if
the retriever returns the wrong documents, even a perfect LLM will generate a wrong
answer. You can evaluate retrieval relevance by:
- Checking if the retrieved documents contain information related to the query topic
- Measuring whether expected keywords or concepts appear in the retrieved context
- Using an LLM-as-judge to rate relevance on a scale

### Answer Faithfulness
Is the generated answer faithful to the retrieved context? A faithful answer only
makes claims that are supported by the provided documents. An unfaithful answer
invents facts, adds details not in the context, or contradicts the source material.
This is the hallucination problem.

### Answer Completeness
Does the answer fully address the question? An answer can be faithful (everything
it says is in the context) but incomplete (it missed important details). For example,
if a customer asks "What SUVs do you have under $40,000?" and the answer only mentions
two of the four qualifying SUVs, it is faithful but incomplete.

Script `03_evaluation.py` demonstrates how to run these evaluations programmatically
against a set of test cases.

---

## Data Freshness

A RAG system is only as good as its knowledge base. If a vehicle is sold but still
listed in the inventory, the chatbot will recommend it to customers. If a promotion
expires but the promotions document is not updated, the chatbot will advertise deals
that no longer exist.

Data freshness monitoring means tracking:

- **When was each document last updated?** Store timestamps as metadata in ChromaDB.
- **How often does the source data change?** Inventory changes daily; policies change
  rarely. Set different staleness thresholds for different document types.
- **Are there gaps in coverage?** If customers keep asking about a topic and the
  retriever returns low-relevance results, you may be missing documents.

In a production system you would build a pipeline that periodically refreshes the
ChromaDB collection from your source-of-truth systems (dealer management system,
CMS, etc.). For this educational project, the key takeaway is: your chatbot is only
as current as the data you feed it.

---

## The RAG Pipeline as a Traceable System

When Phoenix traces a RAG query, you see the full pipeline as a hierarchy of spans:

```
Trace: "What SUVs are available under $40,000?"
  |
  +-- Retriever span (45ms)
  |     Input: "What SUVs are available under $40,000?"
  |     Output: [4 documents retrieved]
  |     Distances: [0.42, 0.58, 0.71, 0.89]
  |
  +-- LLM span (1,230ms)
        Model: meta-llama/Llama-3.3-70B-Instruct-Turbo
        Input tokens: 847
        Output tokens: 256
        Prompt: [system message + context + user question]
        Response: "We have several SUVs under $40,000..."
```

This visibility lets you:

- **Debug retrieval**: If the answer is wrong, check whether the retriever found the
  right documents. If not, the problem is in your embeddings or document chunking.
- **Debug generation**: If the retriever found the right documents but the answer is
  still wrong, the problem is in your prompt or model choice.
- **Optimize latency**: See exactly where time is spent. If the retriever is fast but
  the LLM is slow, consider a faster model or shorter prompts.
- **Track costs**: Sum up token usage across all requests to project monthly API costs.

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A [Together.ai](https://together.ai) API key (free tier available)
- All dependencies from earlier phases plus Phoenix and OpenTelemetry packages

### 1. Install dependencies

```bash
cd phase-6-monitoring-and-evaluation
pip install -r requirements.txt
```

### 2. Configure your environment

Create a `.env` file in the `phase-6-monitoring-and-evaluation/` directory:

```bash
TOGETHER_API_KEY=your-api-key-here
```

Or export it in your terminal:

```bash
export TOGETHER_API_KEY="your-api-key-here"
```

---

## Running the Scripts

### Script 1: Phoenix Setup (`01_phoenix_setup.py`)

Launches the Arize Phoenix dashboard and sends a single traced LLM call so you can
see what tracing looks like. This is your "hello world" for observability.

```bash
python 01_phoenix_setup.py
```

After running, open [http://localhost:6006](http://localhost:6006) in your browser.
You should see one trace with an LLM span showing the prompt, response, token count,
and latency. Press Ctrl+C or Enter to stop.

Requires `TOGETHER_API_KEY`.

### Script 2: Traced RAG Pipeline (`02_tracing_rag.py`)

Builds a full RAG pipeline using LangChain with Phoenix tracing active. Loads all
dealership documents into ChromaDB, constructs a LangChain retrieval chain, and
runs several test queries. Each query generates a full trace in Phoenix showing
the retriever span and LLM span.

```bash
python 02_tracing_rag.py
```

After the queries complete, open [http://localhost:6006](http://localhost:6006) to
explore the traces. Click on any trace to see the retrieval results, the full prompt
sent to the LLM, the response, and timing for each step.

Requires `TOGETHER_API_KEY`.

### Script 3: RAG Evaluation (`03_evaluation.py`)

Runs a set of predefined test cases through the RAG pipeline and evaluates each one
for retrieval relevance and answer completeness. Prints a summary report and saves
detailed results to `evaluation_results.json`.

```bash
python 03_evaluation.py
```

Review the printed report to see which queries performed well and which need
improvement. Open `evaluation_results.json` for the full details including retrieved
documents and scores.

Requires `TOGETHER_API_KEY`.

### Script 4: Monitored Chainlit App (`04_chainlit_app.py`)

The complete chatbot with full observability. Builds on everything from Phase 5
(context engineering, memory, guardrails, mode toggle, model selection) and adds
Phoenix tracing, token counting, and latency tracking for every response.

```bash
chainlit run 04_chainlit_app.py
```

This starts the chatbot at [http://localhost:8000](http://localhost:8000) and the
Phoenix dashboard at [http://localhost:6006](http://localhost:6006). Every conversation
is traced. Each response includes a footer showing token count, latency, and a link
to the Phoenix dashboard.

Requires `TOGETHER_API_KEY`.

---

## What You Learned

After completing this phase, you understand:

- **Observability is not optional** for production LLM applications. Without it you
  are flying blind.
- **Arize Phoenix** provides a local, open-source dashboard for inspecting LLM traces.
- **OpenTelemetry** is the standard protocol for collecting and exporting traces.
- **Automatic instrumentation** via OpenInference means you do not need to manually
  add tracing code to every function.
- **Evaluation** is different from monitoring: monitoring tells you what happened,
  evaluation tells you if it was good.
- **Retrieval relevance, answer faithfulness, and answer completeness** are the three
  dimensions of RAG quality.
- **Data freshness** is a critical concern -- your chatbot is only as current as
  your knowledge base.

This completes the chroma-rag-ce educational project. You have built a RAG system
from scratch: loading documents, embedding them, searching them, building a chatbot
UI, engineering context for better responses, and monitoring the whole system.
