"""
Phase 5 - Script 03: Context Window Visualization
===================================================
Builds a complete RAG context with all four components (system prompt,
retrieved documents, chat history, user query), counts tokens for each
section, and prints a visual breakdown. Then demonstrates what happens
when the context overflows and how a pruning strategy helps.

Run:
    python 03_context_window.py
"""

import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, SystemMessage

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split())

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Context components
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """<role>
You are a senior sales assistant at AutoElite Motors car dealership.
You have deep knowledge of the inventory, financing options, and current promotions.
</role>

<task>
Help customers find the right vehicle by answering their questions accurately
based on the provided context. Guide them through the car-buying process.
</task>

<constraints>
- Only provide information present in the context
- If information is not available, acknowledge it honestly
- Never make up prices, features, or availability
- Do not discuss competitors
- Escalate complex financing questions to the finance department
</constraints>

<output_format>
- Start with a direct answer to the question
- Use bullet points for listing multiple items
- Include specific details (price, year, features) when available
- End with a helpful next step or suggestion
- Keep responses to 2-3 paragraphs maximum
</output_format>"""


def load_documents() -> str:
    """Load sample documents from the data directory to simulate retrieval."""
    docs = []

    # Load inventory
    inventory_path = DATA_DIR / "inventory.csv"
    if inventory_path.exists():
        with open(inventory_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status", "").lower() == "available":
                    doc = (
                        f"{row['year']} {row['make']} {row['model']} {row['trim']}: "
                        f"${row['price']}, {row['condition']}, {row['color']}. "
                        f"Features: {row['features']}."
                    )
                    docs.append(doc)

    # Load promotions
    promo_path = DATA_DIR / "promotions.md"
    if promo_path.exists():
        with open(promo_path, "r") as f:
            docs.append(f.read()[:500])  # Take first 500 chars

    # Load some FAQ
    faq_path = DATA_DIR / "faq.json"
    if faq_path.exists():
        with open(faq_path, "r") as f:
            faqs = json.load(f)
            for faq in faqs[:3]:
                docs.append(f"Q: {faq['question']}\nA: {faq['answer']}")

    return "\n\n".join(docs)


def build_chat_history(num_turns: int = 5) -> list:
    """Generate simulated chat history."""
    exchanges = [
        ("What SUVs do you have?", "We have several SUVs available! The 2024 Honda CR-V EX-L at $38,450, the 2024 Hyundai Tucson SEL at $31,750, the 2024 Ford Bronco Sport Big Bend at $35,890, and the 2024 Chevrolet Equinox RS at $33,200. Would you like details on any of these?"),
        ("Tell me more about the Tucson.", "The 2024 Hyundai Tucson SEL is priced at $31,750 in Amazon Gray. It features a 2.5L engine, 8-inch touchscreen, Hyundai SmartSense safety suite, blind-spot monitoring, and LED headlamps. It is a great value for a family SUV. Would you like to schedule a test drive?"),
        ("What about financing?", "We offer competitive financing through over 20 lenders. For new vehicles, APR starts at 2.9% for 36-month terms. We have terms from 24 to 72 months with no prepayment penalties. Right now, select models qualify for 0% APR for 60 months through our Winter Clearance Event. Shall I run some numbers for you?"),
        ("Do you have any promotions right now?", "Yes! Our Winter Clearance Event offers up to $3,000 off MSRP on select 2024 models, including the Hyundai Tucson. We also have a Trade-In Bonus of $1,500 above KBB value when purchasing a new vehicle. And if you are military or a first responder, there is an additional $750 discount. These can be combined!"),
        ("I have a trade-in. What do I need to bring?", "For a trade-in appraisal, please bring your vehicle title, registration, and all keys and remotes. Our appraisals are based on Kelley Blue Book, NADA guides, and live auction data, and are valid for 7 days. With the current Trade-In Bonus, you will get $1,500 above the standard KBB value."),
    ]
    return exchanges[:num_turns]


USER_QUERY = "I think I want the Tucson. What would my monthly payment be with the trade-in bonus and 60-month financing?"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def print_bar(fraction: float, width: int = 10) -> str:
    """Return a text-based progress bar."""
    filled = int(fraction * width)
    filled = min(filled, width)
    return "#" * filled + "-" * (width - filled)


def visualize_context(context_limit: int, system_tokens: int,
                      docs_tokens: int, history_tokens: int,
                      query_tokens: int, label: str = ""):
    """Print a visual breakdown of context window usage."""

    total_used = system_tokens + docs_tokens + history_tokens + query_tokens
    available = max(0, context_limit - total_used)

    sections = [
        ("System Prompt", system_tokens),
        ("Retrieved Docs", docs_tokens),
        ("Chat History", history_tokens),
        ("User Query", query_tokens),
        ("Available for LLM", available),
    ]

    if label:
        print(f"\n{label}")
    print(f"Context Window Usage ({context_limit} tokens):")
    print("=" * 64)

    for name, tokens in sections:
        pct = (tokens / context_limit) * 100 if context_limit > 0 else 0
        bar = print_bar(tokens / context_limit if context_limit > 0 else 0)
        print(f"| {name:<20} | {tokens:>5} tokens | {pct:>5.1f}% | {bar} |")

    print("=" * 64)

    if total_used > context_limit:
        overflow = total_used - context_limit
        print(f"*** OVERFLOW: {overflow} tokens over the limit! ***")
    else:
        print(f"Total used: {total_used} tokens ({(total_used/context_limit)*100:.1f}%)")


# ---------------------------------------------------------------------------
# Pruning strategy: summarize old history
# ---------------------------------------------------------------------------

def summarize_history(exchanges: list, llm) -> str:
    """Use the LLM to summarize a list of conversation exchanges."""
    text = ""
    for user_msg, ai_msg in exchanges:
        text += f"Customer: {user_msg}\nAssistant: {ai_msg}\n\n"

    prompt = (
        "Summarize this car dealership conversation concisely. "
        "Preserve the customer's preferences, budget, vehicles discussed, "
        "and key decisions. Write 2-3 sentences.\n\n" + text
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main():
    print("Phase 5 - Context Window Visualization")
    print("=" * 64)

    # --- Step 1: Build the full context ---
    print("\nStep 1: Building a complete RAG context...")

    retrieved_docs = load_documents()
    chat_history = build_chat_history(5)

    # Count tokens for each section
    system_tokens = count_tokens(SYSTEM_PROMPT)
    docs_tokens = count_tokens(retrieved_docs)

    history_text = ""
    for user_msg, ai_msg in chat_history:
        history_text += f"Customer: {user_msg}\nAssistant: {ai_msg}\n"
    history_tokens = count_tokens(history_text)

    query_tokens = count_tokens(USER_QUERY)

    print(f"\nToken counts:")
    print(f"  System prompt:    {system_tokens:>5} tokens")
    print(f"  Retrieved docs:   {docs_tokens:>5} tokens")
    print(f"  Chat history:     {history_tokens:>5} tokens")
    print(f"  User query:       {query_tokens:>5} tokens")

    # --- Step 2: Visualize with a 4096-token window ---
    CONTEXT_LIMIT = 4096
    visualize_context(
        CONTEXT_LIMIT, system_tokens, docs_tokens,
        history_tokens, query_tokens,
        label="SCENARIO 1: Normal context (4096-token window)"
    )

    # --- Step 3: Simulate overflow ---
    print("\n" + "-" * 64)
    print("Step 2: Simulating context overflow...")
    print("Adding more conversation history to push past the limit.\n")

    # Double the history by repeating it
    extended_history = chat_history * 3  # 15 turns
    extended_text = ""
    for user_msg, ai_msg in extended_history:
        extended_text += f"Customer: {user_msg}\nAssistant: {ai_msg}\n"
    extended_history_tokens = count_tokens(extended_text)

    visualize_context(
        CONTEXT_LIMIT, system_tokens, docs_tokens,
        extended_history_tokens, query_tokens,
        label="SCENARIO 2: Extended conversation (context overflow!)"
    )

    # --- Step 4: Apply pruning strategy ---
    print("\n" + "-" * 64)
    print("Step 3: Applying pruning strategy...")
    print("Summarizing older conversation history to reclaim tokens.\n")

    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=150,
    )

    # Summarize the first 12 turns, keep the last 3
    old_exchanges = extended_history[:12]
    recent_exchanges = extended_history[12:]

    summary = summarize_history(old_exchanges, llm)
    print(f"Summary of older turns:\n  \"{summary}\"\n")

    # Recalculate tokens
    summary_tokens = count_tokens(summary)
    recent_text = ""
    for user_msg, ai_msg in recent_exchanges:
        recent_text += f"Customer: {user_msg}\nAssistant: {ai_msg}\n"
    recent_tokens = count_tokens(recent_text)

    pruned_history_tokens = summary_tokens + recent_tokens

    visualize_context(
        CONTEXT_LIMIT, system_tokens, docs_tokens,
        pruned_history_tokens, query_tokens,
        label="SCENARIO 3: After pruning (summary + recent history)"
    )

    # --- Summary ---
    print("\n" + "=" * 64)
    print("  KEY TAKEAWAYS")
    print("=" * 64)
    print(f"""
1. Every component competes for space in the context window.
   System prompt ({system_tokens} tokens) + retrieved docs ({docs_tokens} tokens)
   are relatively fixed per query; history grows over time.

2. Overflow is silent and dangerous. If you exceed the limit,
   the model either truncates (losing your system prompt) or errors out.
   Neither is acceptable in production.

3. Summarization is an effective pruning strategy:
   - 15-turn history: {extended_history_tokens} tokens
   - After summarization: {pruned_history_tokens} tokens
   - Tokens saved: {extended_history_tokens - pruned_history_tokens}

4. A practical token budget for a 4096-token window:
   - System Prompt:  ~300 tokens (7%)
   - Retrieved Docs: ~1200 tokens (30%)
   - Chat History:   ~800 tokens (20%)
   - User Query:     ~100 tokens (2%)
   - Response Space:  ~1600 tokens (40%)

5. Always monitor token usage. The visualization above makes it
   easy to spot when a section is consuming too much of the budget.
""")


if __name__ == "__main__":
    main()
