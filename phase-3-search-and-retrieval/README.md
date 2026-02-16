# Phase 3: Search and Retrieval

This phase dives into the core of any RAG system: how you search your vector database
and retrieve documents that are relevant to a user's question. Retrieval quality is
the single most important factor in RAG performance -- a perfect language model cannot
compensate for irrelevant context, and a mediocre model can still give great answers
when it receives the right documents.

---

## Concepts Covered

### How Semantic Search Works

Traditional keyword search (like a SQL `LIKE '%toyota%'` query) matches exact words.
If a customer asks "fuel efficient family car," keyword search will not find a document
about the "Toyota Camry with adaptive cruise control" because none of those exact words
overlap. This is the vocabulary mismatch problem.

Semantic search solves this by converting text into dense numerical vectors (embeddings)
that capture meaning rather than exact words. Here is the intuition:

1. **Embedding**: Each piece of text is transformed into a vector of numbers (e.g., 384
   dimensions). Think of this as placing the text at a specific point in a
   high-dimensional "meaning space."

2. **Proximity equals similarity**: Texts with similar meanings end up near each other
   in this space. "fuel efficient family car" and "Toyota Camry LE with great gas
   mileage" will be close together, even though they share no words.

3. **Distance metrics**: When you query, the database calculates the distance between
   your query vector and every document vector. Common metrics include:
   - **L2 (Euclidean) distance**: Straight-line distance between two points. Lower
     values mean more similar. ChromaDB uses this by default.
   - **Cosine similarity**: Measures the angle between two vectors. Values range from
     -1 to 1, where 1 means identical direction. Often converted to cosine distance
     (1 - cosine_similarity), so lower is still more similar.
   - **Inner product (dot product)**: Related to cosine similarity but also accounts
     for vector magnitude.

4. **Retrieval**: The database returns the k nearest documents, ranked by distance.

### Keyword/Sparse Search vs. Semantic/Dense Search

| Aspect | Keyword (Sparse) Search | Semantic (Dense) Search |
|--------|------------------------|------------------------|
| How it works | Exact or fuzzy word matching (TF-IDF, BM25) | Embedding similarity in vector space |
| Strengths | Precise for known terms, product codes, names | Understands intent, synonyms, paraphrases |
| Weaknesses | Misses synonyms and paraphrased queries | Can miss exact terms; less precise for IDs |
| Example win | "F-150 XLT" finds exactly that truck | "good truck for towing" finds the F-150 |
| Example loss | "fuel efficient family car" finds nothing | "stock number A1234" might return wrong items |

In practice, the best RAG systems often combine both approaches (hybrid search). For
this phase, we focus on semantic search with metadata filtering as a complement.

### Metadata Filtering in ChromaDB

ChromaDB supports filtering at query time using two mechanisms:

- **`where` clause**: Filters on metadata fields attached to documents.
  ```python
  collection.query(
      query_texts=["family car"],
      where={"make": "Toyota"},   # Only search Toyota vehicles
      n_results=3
  )
  ```

- **`where_document` clause**: Filters on the document text content itself.
  ```python
  collection.query(
      query_texts=["SUV"],
      where_document={"$contains": "leather"},  # Must mention "leather"
      n_results=3
  )
  ```

- **Operators**: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin` for
  metadata; `$contains`, `$not_contains` for document content.

- **Logical combinators**: `$and`, `$or` for combining multiple conditions.
  ```python
  where={"$and": [{"year": {"$gte": 2024}}, {"price": {"$lte": 35000}}]}
  ```

### Internal vs. External Data (Access Levels)

A real dealership RAG system serves two audiences:

- **External (customer-facing)**: Inventory listings, promotions, FAQ answers,
  dealership policies. This is information any customer should be able to access.

- **Internal (employee-facing)**: Customer complaints, internal service notes,
  financial details, employee procedures. This data must never leak to customers.

By tagging each document with an `access_level` metadata field ("external" or
"internal"), we can filter at query time to enforce data boundaries. A customer-facing
chatbot filters to `where={"access_level": "external"}`, while an employee dashboard
can access everything.

### Why Retrieval Quality Matters Most

Consider the RAG pipeline: Query -> Retrieve Documents -> Generate Answer.

If retrieval returns irrelevant documents, the LLM either hallucinates from bad context
or correctly says "I don't know" -- either way, the user gets a poor experience. Studies
consistently show that improving retrieval quality (better embeddings, better chunking,
better filtering) has a much larger impact on RAG accuracy than swapping to a more
powerful LLM.

This phase teaches you to:
- Understand how embedding distances relate to relevance
- Use metadata filters to narrow the search space
- Combine semantic search with structured filters for precision
- Enforce access control through metadata boundaries

---

## Scripts

### 01_embedding_models.py
Demonstrates how ChromaDB's default embeddings work and what distance scores mean.
Creates a collection of car-related documents, runs several test queries, and prints
a formatted results table showing documents ranked by distance.

```bash
python 01_embedding_models.py
```

No API key required -- uses ChromaDB's built-in default embedding model.

### 02_search_modalities.py
Compares four different search approaches in ChromaDB using the dealership data files:
1. Pure semantic search
2. Document content filtering (`where_document`)
3. Metadata filtering (`where`)
4. Combined semantic + metadata search

```bash
python 02_search_modalities.py
```

No API key required.

### 03_metadata_filtering.py
Deep dive into metadata filtering with the full dealership dataset, including both
customer-facing inventory and internal complaint records. Demonstrates numeric range
filters, category filters, access-level filtering, and combined queries.

```bash
python 03_metadata_filtering.py
```

No API key required.

### 04_retrieval_rag.py
Complete RAG pipeline with access-controlled retrieval. Loads all dealership data into
ChromaDB with access-level metadata, retrieves relevant documents based on user type,
and sends them as context to a Together.ai LLM for answer generation.

```bash
# Set your API key
export TOGETHER_API_KEY="your-key-here"

# Optionally set a different model
export TOGETHER_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

python 04_retrieval_rag.py
```

Features an interactive CLI where you choose customer or employee mode, then ask
questions. Type "switch" to change mode or "quit" to exit.

---

## Data Files

All data files live in the `data/` directory:

| File | Content | Access Level |
|------|---------|-------------|
| `inventory.csv` | Vehicle listings with make, model, year, price, etc. | external |
| `policies.txt` | Dealership warranty, return, financing, and service policies | external |
| `promotions.md` | Current deals, discounts, and special offers | external |
| `faq.json` | Frequently asked questions with answers | external |
| `complaints.csv` | Customer complaints with status and resolution details | internal |

---

## Key Takeaways

1. Semantic search finds relevant documents even when the query uses completely
   different words than the documents contain.

2. Distance scores tell you how confident the match is -- always inspect them to
   understand retrieval quality.

3. Metadata filtering narrows the search space before semantic matching, giving you
   both precision (filtering) and recall (semantic similarity).

4. Access-level filtering is a simple but effective pattern for controlling what
   information different user types can see in a RAG system.

5. Retrieval is the bottleneck. Invest in better embeddings, better chunking, and
   better filtering before reaching for a bigger LLM.
