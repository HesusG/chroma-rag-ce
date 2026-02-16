# ChromaDB RAG Chatbot — Car Dealership (Context Engineering Edition)

> An educational project that progressively builds a RAG chatbot for a car dealership using ChromaDB, LangChain, and Together.ai. Companion project for the [DeepLearning.AI Context Engineering](https://www.deeplearning.ai/short-courses/context-engineering/) course and inspired by this [context engineering explainer video](https://www.youtube.com/watch?v=J2sUCSMRtHY).

## What You'll Build

A full-featured RAG chatbot for **AutoElite Motors** that:
- Answers customer questions about inventory, financing, and promotions
- Helps employees access complaints, support tickets, and training materials
- Uses context engineering to deliver accurate, well-structured responses
- Includes full observability with tracing and evaluation

## Learning Path

| Phase | Topic | What You'll Learn |
|-------|-------|-------------------|
| [Phase 1](phase-1-foundations/) | Foundations | RAG basics, ChromaDB, embeddings, your first query |
| [Phase 2](phase-2-document-loading/) | Document Loading | Multiple formats (CSV, JSON, MD), chunking strategies |
| [Phase 3](phase-3-search-and-retrieval/) | Search & Retrieval | Semantic search, metadata filtering, hybrid search |
| [Phase 4](phase-4-chatbot-ui/) | Chatbot UI | Chainlit web interface, PDF loading, customer/employee modes |
| [Phase 5](phase-5-context-engineering/) | Context Engineering | System prompts, memory, context window management |
| [Phase 6](phase-6-monitoring-and-evaluation/) | Monitoring & Evaluation | Arize Phoenix tracing, RAG evaluation metrics |

## Prerequisites

- Python 3.10+
- A [Together.ai](https://together.ai) API key (free tier available)
- Basic Python knowledge

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/chroma-rag-ce.git
cd chroma-rag-ce

# Start with Phase 1
cd phase-1-foundations
pip install -r requirements.txt

# Set your API key
export TOGETHER_API_KEY="your-key-here"

# Run your first RAG query
python 01_hello_chroma.py
```

## Course Companion

This project maps to the [DeepLearning.AI Context Engineering](https://www.deeplearning.ai/short-courses/context-engineering/) course:

| Course Module | Phase(s) | Topics Covered |
|---------------|----------|----------------|
| Module 1: Intro to Context Engineering | Phase 1, 5 | RAG basics, context engineering concepts, failure modes |
| Module 2: Building a Context Engineering System | Phase 2, 3, 4 | Search modalities, chunking, tools, data loading |
| Module 3: Context Management | Phase 5 | Pruning, highlighting, summarization, memory |
| Module 4: Offloading | Phase 5, 6 | Memory stores, conversation persistence, evaluation |

## Tech Stack

| Tool | Purpose |
|------|---------|
| [ChromaDB](https://www.trychroma.com/) | Vector database for storing and retrieving document embeddings |
| [LangChain](https://www.langchain.com/) | Framework for building RAG pipelines |
| [Together.ai](https://together.ai) | LLM and embedding model provider (free models available) |
| [Chainlit](https://chainlit.io/) | Chat UI framework (Phase 4+) |
| [Arize Phoenix](https://phoenix.arize.com/) | LLM observability and evaluation (Phase 6) |

## Available Models

All phases use Together.ai. These models are available in the UI dropdown (Phase 4+):

| Model | Model ID | Notes |
|-------|----------|-------|
| Llama 3.3 70B (Free) | `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free` | Default — best free option |
| DeepSeek R1 70B (Free) | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free` | Reasoning model, verbose output |
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct-Turbo` | Fast, uses free credits |
| Llama 3.1 8B | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | Lightweight, uses free credits |

## Project Structure

```
chroma-rag-ce/
├── README.md
├── .gitignore
├── phase-1-foundations/          # RAG basics with ChromaDB
├── phase-2-document-loading/     # Multi-format document loading & chunking
├── phase-3-search-and-retrieval/ # Advanced search & metadata filtering
├── phase-4-chatbot-ui/           # Chainlit web interface
├── phase-5-context-engineering/  # Prompt engineering & memory
└── phase-6-monitoring-and-evaluation/ # Phoenix tracing & evaluation
```

Each phase is **self-contained** — it has its own `requirements.txt`, data files, and README with explanations. Work through them in order for the best learning experience.

## License

This project is for educational purposes. Feel free to use it to learn about RAG and context engineering.
