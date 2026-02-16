# Phase 2: Document Loading & Chunking

> Learn how to load documents from multiple formats and split them into chunks that are optimized for retrieval.

## Why Document Format Matters for RAG

In a real-world RAG system, your knowledge base is rarely a single plain text file. It is a mix of CSVs (inventory data), JSON (APIs, FAQs), Markdown (documentation, promotions), plain text (policies, manuals), and more. Each format has its own structure:

- **CSV** rows are naturally independent records (one row = one car in our inventory).
- **JSON** entries may have nested fields where only some keys contain the answer.
- **Markdown** has headings, lists, and sections that carry semantic meaning.
- **Plain text** has no formal structure -- it relies on headings and whitespace.

A good RAG pipeline must understand these differences and load each format in a way that preserves its meaning. That is what **document loaders** do.

---

## LangChain Document Loaders

LangChain provides a loader for nearly every common format. Each loader reads a file and returns a list of `Document` objects, where every document has two things:

1. **`page_content`** -- the text that will be embedded and searched.
2. **`metadata`** -- a dictionary of information about where the text came from (file name, row number, etc.).

### Loaders Used in This Phase

| Loader | Format | How It Works |
|--------|--------|--------------|
| `TextLoader` | `.txt` | Loads the entire file as a single document. |
| `CSVLoader` | `.csv` | Loads each row as a separate document. Column values become the content. |
| `UnstructuredMarkdownLoader` | `.md` | Parses Markdown structure (headings, lists) into documents. Requires the `unstructured` package. Falls back to `TextLoader` if unavailable. |
| `JSONLoader` | `.json` | Uses a `jq`-style schema to extract specific fields from JSON. You control which field becomes the content and which become metadata. |

### Example: What a Loaded Document Looks Like

```python
from langchain_community.document_loaders import TextLoader

docs = TextLoader("data/policies.txt").load()
print(docs[0].page_content[:100])   # First 100 characters of the text
print(docs[0].metadata)              # {'source': 'data/policies.txt'}
```

Every loader produces the same `Document` shape, so the rest of your pipeline (chunking, embedding, retrieval) does not need to care what format the original file was.

---

## What Is Chunking and Why Is It Critical?

**Chunking** is the process of splitting a document into smaller pieces before storing them in a vector database. It is one of the most impactful decisions in a RAG pipeline.

### Why not just embed the whole document?

1. **Embedding models have token limits.** Most models accept 512 or 8,192 tokens. A 10-page policy document will not fit.
2. **Retrieval is about precision.** If a user asks "What is the return policy?", you want to retrieve *just* the return policy section -- not the entire 80-line policies file. Smaller, focused chunks lead to more relevant results.
3. **LLMs have context windows.** Even if you could embed a whole document, stuffing an entire document into the LLM prompt wastes context space and can confuse the model.

### The Goldilocks Problem

```
Chunk too BIG:
  "Here are all 6 policies for the dealership..."
  -> The embedding is a vague average of everything.
  -> Retrieved chunk contains mostly irrelevant information.
  -> LLM has to find the needle in the haystack.

Chunk too SMALL:
  "- Bumper-to-bumper: 3 years or 36,000 miles"
  -> The embedding is very specific but lacks context.
  -> The LLM does not know this is about WARRANTY policy.
  -> Answer may be incomplete or misleading.

Chunk JUST RIGHT:
  "WARRANTY POLICY\nAll new vehicles come with the full manufacturer's
   warranty. Coverage includes:\n- Bumper-to-bumper: 3 years or 36,000
   miles\n- Powertrain: 5 years or 60,000 miles..."
  -> The embedding captures the topic (warranty) and the details.
  -> Retrieved chunk gives the LLM enough context to answer accurately.
```

---

## Chunking Strategies

### RecursiveCharacterTextSplitter (Recommended Default)

This splitter tries to split on **natural boundaries** in this order:

1. Double newlines (`\n\n`) -- paragraph breaks
2. Single newlines (`\n`) -- line breaks
3. Spaces (` `) -- word boundaries
4. Characters -- last resort

It "recurses" through these separators, using the most meaningful split point that keeps each chunk under the size limit. This preserves paragraphs and sentences as much as possible.

### CharacterTextSplitter

This splitter uses a **single separator** (default: `\n\n`). If a chunk is still too big after splitting on that separator, it stays oversized. It is simpler but less adaptive.

### Key Parameters

| Parameter | What It Controls |
|-----------|-----------------|
| `chunk_size` | Maximum number of characters per chunk. |
| `chunk_overlap` | Number of characters shared between consecutive chunks. |

---

## Chunk Size and Overlap -- How They Affect Retrieval Quality

### Visual Example

Imagine this simplified text:

```
AAAA AAAA AAAA BBBB BBBB BBBB CCCC CCCC CCCC
```

**chunk_size=12, chunk_overlap=0 (no overlap):**
```
[AAAA AAAA AA] [AA BBBB BBBB] [BBBB CCCC CC] [CC CCCC]
                ^               ^
          Info about B is      Info about B is
          split across         split across
          two chunks           two chunks
```

**chunk_size=12, chunk_overlap=4:**
```
[AAAA AAAA AA] [A AAAA BBBB] [BBBB BBBB CC] [CCCC CCCC]
                    ^^^^          ^^^^
               Overlap means   Overlap means
               context from    context from
               chunk 1 bleeds  chunk 2 bleeds
               into chunk 2    into chunk 3
```

Overlap acts as a **safety net**: if a split happens in the middle of an important passage, the overlapping portion ensures the next chunk also contains that context. A common rule of thumb is to set overlap to **10-20%** of chunk size.

### Guidelines for Choosing Chunk Size

| Chunk Size | Best For | Trade-off |
|------------|----------|-----------|
| 100-200 chars | Short factual lookups (FAQ, definitions) | May lose surrounding context |
| 300-500 chars | General-purpose RAG (policies, articles) | Good balance of precision and context |
| 500-1000 chars | Long-form content (manuals, legal docs) | Chunks may contain mixed topics |
| 1000+ chars | Summarization tasks | Less precise retrieval |

---

## How Metadata Is Preserved Through Loading and Chunking

When a loader creates a document, it attaches metadata (e.g., `source`, `row` for CSVs). When that document is then chunked, **each chunk inherits the metadata from its parent document**. This means you can always trace a chunk back to its source file and location.

```
Original Document:
  page_content: "WARRANTY POLICY ... SERVICE & MAINTENANCE ..."
  metadata: {"source": "policies.txt"}

After Chunking:
  Chunk 1: page_content: "WARRANTY POLICY ..."
            metadata: {"source": "policies.txt"}    <-- preserved!
  Chunk 2: page_content: "RETURN POLICY ..."
            metadata: {"source": "policies.txt"}    <-- preserved!
  Chunk 3: page_content: "SERVICE & MAINTENANCE ..."
            metadata: {"source": "policies.txt"}    <-- preserved!
```

You can also **add custom metadata** during loading (like document type or category) and it will flow through the entire pipeline into ChromaDB, where it can be used for filtered searches.

---

## Scripts in This Phase

| Script | What It Does |
|--------|--------------|
| `01_document_loaders.py` | Loads documents from 4 different formats (TXT, CSV, MD, JSON) and inspects the results. |
| `02_chunking_strategies.py` | Compares chunking strategies and chunk sizes on the same document. |
| `03_multi_doc_rag.py` | Full RAG pipeline that loads, chunks, and stores all document types, then answers questions with source attribution. |

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key (needed only for script 03)
export TOGETHER_API_KEY="your-key-here"

# 3. Run each script
python 01_document_loaders.py
python 02_chunking_strategies.py
python 03_multi_doc_rag.py
```

Scripts 01 and 02 require no API key -- they demonstrate loading and chunking locally. Script 03 connects to Together.ai for the LLM and runs an interactive question-answering loop.

---

## Key Takeaways

1. **Use the right loader for each format.** CSVLoader gives you one document per row. JSONLoader lets you extract specific fields. TextLoader is the simplest fallback.
2. **RecursiveCharacterTextSplitter is the best default.** It splits on natural boundaries and adapts to different content structures.
3. **Chunk size is a trade-off.** Smaller chunks are more precise but may lack context. Larger chunks have more context but may dilute relevance. Start with 300-500 characters and adjust based on your retrieval quality.
4. **Overlap is your safety net.** Set it to 10-20% of chunk size to avoid losing information at split boundaries.
5. **Metadata flows through the pipeline.** Use it to filter and attribute results back to their source.
