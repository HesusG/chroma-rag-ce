"""
Phase 5 - Script 02: Conversation Memory
==========================================
Implements three conversation memory strategies from scratch and
simulates a multi-turn conversation to compare how each handles
long-term context retention.

The three strategies:
  1. Buffer Memory:  Store every message. Simple but unbounded.
  2. Window Memory:  Keep only the last N messages.
  3. Summary Memory: Summarize older messages, keep recent ones.

Run:
    python 02_conversation_memory.py
"""

import os
from dotenv import load_dotenv
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
# System prompt (shared across all memory strategies)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful sales assistant at AutoElite Motors.
Answer customer questions about vehicles, financing, and promotions.
Keep responses concise (2-3 sentences). Use the conversation context
to give personalized answers."""

# ---------------------------------------------------------------------------
# Simulated multi-turn conversation
# ---------------------------------------------------------------------------

CONVERSATION_TURNS = [
    "Hi! I'm looking for a family SUV. My budget is around $35,000.",
    "Do you have any Honda models? I've always driven Hondas.",
    "What financing options are available? I'd like a 60-month term.",
    "Actually, do you also have any trucks? My husband might want one.",
    "What's the trade-in process like? We have a 2019 Honda Pilot to trade in.",
    "Going back to the beginning -- what was my budget again, and which SUV did you first recommend for me?",
]

# ---------------------------------------------------------------------------
# Memory Strategy 1: Buffer Memory
# ---------------------------------------------------------------------------

class BufferMemory:
    """Store every message. No pruning, no summarization."""

    def __init__(self):
        self.messages = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_ai_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self):
        """Return all messages as LangChain message objects."""
        result = []
        for msg in self.messages:
            if msg["role"] == "user":
                result.append(HumanMessage(content=msg["content"]))
            else:
                result.append(AIMessage(content=msg["content"]))
        return result

    def get_token_count(self) -> int:
        return sum(count_tokens(m["content"]) for m in self.messages)

    def get_message_count(self) -> int:
        return len(self.messages)

    def describe(self) -> str:
        return (
            f"Buffer Memory: {self.get_message_count()} messages, "
            f"{self.get_token_count()} tokens"
        )


# ---------------------------------------------------------------------------
# Memory Strategy 2: Window Memory
# ---------------------------------------------------------------------------

class WindowMemory:
    """Keep only the last N messages. Older ones are dropped."""

    def __init__(self, window_size: int = 6):
        self.messages = []
        self.window_size = window_size  # number of individual messages to keep

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_ai_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self):
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_messages(self):
        result = []
        for msg in self.messages:
            if msg["role"] == "user":
                result.append(HumanMessage(content=msg["content"]))
            else:
                result.append(AIMessage(content=msg["content"]))
        return result

    def get_token_count(self) -> int:
        return sum(count_tokens(m["content"]) for m in self.messages)

    def get_message_count(self) -> int:
        return len(self.messages)

    def describe(self) -> str:
        return (
            f"Window Memory (last {self.window_size}): "
            f"{self.get_message_count()} messages, "
            f"{self.get_token_count()} tokens"
        )


# ---------------------------------------------------------------------------
# Memory Strategy 3: Summary Memory
# ---------------------------------------------------------------------------

class SummaryMemory:
    """Summarize older messages when history exceeds a threshold.
    Keep the summary plus recent messages."""

    def __init__(self, llm, max_messages_before_summary: int = 6):
        self.messages = []
        self.summary = ""
        self.llm = llm
        self.max_messages = max_messages_before_summary

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._maybe_summarize()

    def add_ai_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._maybe_summarize()

    def _maybe_summarize(self):
        """If messages exceed threshold, summarize the older half."""
        if len(self.messages) > self.max_messages:
            # Take the older messages to summarize
            to_summarize = self.messages[:-4]  # Keep the last 4 messages
            keep = self.messages[-4:]

            # Build a text representation of messages to summarize
            conversation_text = ""
            if self.summary:
                conversation_text += f"Previous summary: {self.summary}\n\n"
            for msg in to_summarize:
                role = "Customer" if msg["role"] == "user" else "Assistant"
                conversation_text += f"{role}: {msg['content']}\n"

            # Ask the LLM to summarize
            summary_prompt = (
                "Summarize the following conversation concisely. "
                "Preserve key facts: customer preferences, budget, "
                "vehicle interests, and any decisions made. "
                "Write 2-3 sentences maximum.\n\n"
                f"{conversation_text}"
            )

            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            self.summary = response.content.strip()
            self.messages = keep

    def get_messages(self):
        """Return summary (if any) plus recent messages."""
        result = []
        if self.summary:
            result.append(SystemMessage(
                content=f"Summary of earlier conversation: {self.summary}"
            ))
        for msg in self.messages:
            if msg["role"] == "user":
                result.append(HumanMessage(content=msg["content"]))
            else:
                result.append(AIMessage(content=msg["content"]))
        return result

    def get_token_count(self) -> int:
        total = sum(count_tokens(m["content"]) for m in self.messages)
        if self.summary:
            total += count_tokens(self.summary)
        return total

    def get_message_count(self) -> int:
        count = len(self.messages)
        if self.summary:
            count += 1  # count the summary as one "message"
        return count

    def describe(self) -> str:
        summary_info = ""
        if self.summary:
            summary_info = f" (summary: {count_tokens(self.summary)} tokens)"
        return (
            f"Summary Memory: {self.get_message_count()} messages, "
            f"{self.get_token_count()} tokens{summary_info}"
        )


# ---------------------------------------------------------------------------
# Run the simulation
# ---------------------------------------------------------------------------

def simulate_conversation(strategy_name: str, memory, llm):
    """Run the simulated conversation using the given memory strategy."""

    print(f"\n{'=' * 70}")
    print(f"  STRATEGY: {strategy_name}")
    print(f"{'=' * 70}")

    for i, user_msg in enumerate(CONVERSATION_TURNS, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Customer: {user_msg}")

        # Add user message to memory
        memory.add_user_message(user_msg)

        # Build the full message list: system prompt + memory + current context
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(memory.get_messages())

        # Get LLM response
        response = llm.invoke(messages)
        assistant_msg = response.content.strip()

        # Add assistant response to memory
        memory.add_ai_message(assistant_msg)

        print(f"Assistant: {assistant_msg}")
        print(f"[Memory state: {memory.describe()}]")

    # Final state
    print(f"\n--- Final Memory State ---")
    print(f"{memory.describe()}")

    # Show what's actually in memory
    print("\nMessages in memory:")
    for msg in memory.get_messages():
        role = type(msg).__name__.replace("Message", "")
        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"  [{role}] {content_preview}")


def main():
    print("Phase 5 - Conversation Memory Comparison")
    print("=" * 70)
    print("We will simulate the same 6-turn conversation with three")
    print("different memory strategies and compare the results.")
    print()
    print("Pay special attention to Turn 6, where the customer asks")
    print("about information from Turn 1. Which strategy remembers?")

    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature=0.3,
        max_tokens=256,
    )

    # Strategy 1: Buffer Memory
    buffer_mem = BufferMemory()
    simulate_conversation("Buffer Memory (keep everything)", buffer_mem, llm)

    # Strategy 2: Window Memory (keep last 6 messages = 3 exchanges)
    window_mem = WindowMemory(window_size=6)
    simulate_conversation("Window Memory (last 6 messages)", window_mem, llm)

    # Strategy 3: Summary Memory
    summary_mem = SummaryMemory(llm=llm, max_messages_before_summary=6)
    simulate_conversation("Summary Memory (summarize older turns)", summary_mem, llm)

    # Comparison
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"""
Results for Turn 6 ("What was my budget and which SUV did you first recommend?"):

1. Buffer Memory:
   - Has all messages: SHOULD remember the budget ($35k) and first SUV recommendation.
   - Final state: {buffer_mem.describe()}

2. Window Memory:
   - Only has the last 6 messages (Turns 4-6): CANNOT remember Turn 1.
   - The budget and first recommendation are outside the window.
   - Final state: {window_mem.describe()}

3. Summary Memory:
   - Has a summary of earlier turns plus recent messages.
   - The summary SHOULD contain the budget and SUV preference.
   - Final state: {summary_mem.describe()}

Key takeaways:
- Buffer memory gives perfect recall but grows without bound.
- Window memory is predictable and bounded but loses old context.
- Summary memory balances both: bounded size with key facts preserved.
- The right choice depends on your application: short interactions
  favor window memory; long consultations favor summary memory.
""")


if __name__ == "__main__":
    main()
