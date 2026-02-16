# Phase 5: Context Engineering

## Overview

Phase 5 moves beyond basic RAG into **context engineering** -- the art and science of filling the context window with the right information at the right time. While earlier phases focused on retrieval (getting relevant documents), this phase focuses on everything else that goes into the prompt: how you structure system instructions, how you manage conversation memory, and how you monitor and control token usage across the entire context window.

Context engineering is what separates a demo from a production-quality AI application. A well-engineered context window produces consistent, accurate, and well-formatted responses. A poorly managed one leads to hallucinations, forgotten instructions, and degraded performance over time.

---

## What Is Context Engineering?

Context engineering is the discipline of designing, assembling, and managing the full content of the context window sent to a language model. Every time you call an LLM, you are sending a single block of text (the context window) that includes everything the model needs to generate a good response. Context engineering is about making deliberate choices about what goes into that window.

### The Context Window: Four Components

Every RAG interaction assembles these four components into a single prompt:

```
+-------------------------------------------------------------+
|  1. SYSTEM PROMPT                                           |
|     Role, task, constraints, output format                  |
+-------------------------------------------------------------+
|  2. RETRIEVED DOCUMENTS                                     |
|     Relevant chunks from your vector database               |
+-------------------------------------------------------------+
|  3. CONVERSATION HISTORY                                    |
|     Previous turns of the conversation                      |
+-------------------------------------------------------------+
|  4. USER QUERY                                              |
|     The current question or instruction                     |
+-------------------------------------------------------------+
|  [ Available space for the model's response ]               |
+-------------------------------------------------------------+
```

Each component competes for space in a fixed-size window (e.g., 4096 or 8192 tokens). Context engineering is about allocating that space wisely.

---

## System Prompts

The system prompt sets the foundation for every response. A vague prompt produces vague results. A structured prompt produces structured results.

### Progression of System Prompt Quality

**Level 1 -- Basic:**
```
You are a helpful assistant.
```
This gives the model almost no guidance. Responses will be generic and inconsistent.

**Level 2 -- Structured:**
```
You are a senior sales assistant at AutoElite Motors.

Your role is to help customers find the right vehicle...

Guidelines:
- Always be professional and friendly
- Only provide information based on the context provided
...

Response format:
- Keep responses concise
- Use bullet points for listing options
```
Better. The model now has role clarity and formatting expectations.

**Level 3 -- XML-Tagged:**
```xml
<role>
You are a senior sales assistant at AutoElite Motors.
</role>

<task>
Help customers find the right vehicle by answering their questions
accurately based on the provided context.
</task>

<constraints>
- Only provide information present in the context
- Never make up prices, features, or availability
</constraints>

<output_format>
- Start with a direct answer to the question
- Use bullet points for listing multiple items
- Keep responses to 2-3 paragraphs maximum
</output_format>
```

XML tags give the model explicit structure to parse. Research and practice show that models follow instructions more reliably when the prompt is organized with clear delimiters. The tags act as anchors that the model can reference during generation.

### Why XML Tags Work

- They create unambiguous section boundaries
- The model can "look back" at specific sections (e.g., checking `<constraints>` before generating)
- They reduce instruction drift -- the model is less likely to forget a constraint buried in a paragraph
- They make prompts easier for humans to read, review, and maintain

---

## Conversation Memory

In a multi-turn conversation, you need to carry forward context from earlier turns. But conversation history grows with every exchange, consuming more of your token budget. There are three common approaches:

### 1. Buffer Memory (Store Everything)

Store every message in full. Simple and complete, but grows without bound.

```
Turn 1: User asks about SUVs        ->  stored
Turn 2: Assistant lists 4 SUVs      ->  stored
Turn 3: User asks about financing   ->  stored
Turn 4: Assistant explains APR       ->  stored
Turn 5: User asks about trade-ins   ->  stored
...
Turn 20: Context window is full, instructions are pushed out
```

**Pros:** Perfect recall of all context.
**Cons:** Eventually overflows the context window. System prompt and retrieval get crowded out.

### 2. Window Memory (Keep Last N Turns)

Only keep the most recent N message pairs. Older messages are dropped.

```
Window size = 3 (keep last 3 exchanges)

Turn 1: [dropped]
Turn 2: [dropped]
Turn 3: User asks about financing   ->  kept
Turn 4: Assistant explains APR       ->  kept
Turn 5: User asks about trade-ins   ->  kept
```

**Pros:** Bounded token usage. Simple to implement.
**Cons:** The model forgets everything before the window. If the user references Turn 1, the model has no context.

### 3. Summary Memory (Compress Old Turns)

When conversation history exceeds a threshold, use the LLM itself to summarize older turns into a compact paragraph. Keep the summary plus recent turns.

```
[Summary of Turns 1-8: "Customer is interested in SUVs under $40k,
asked about the CR-V and Tucson, prefers Honda, has a trade-in
worth approximately $18k, and is interested in 60-month financing."]

Turn 9: User asks about Honda warranty  ->  kept in full
Turn 10: Assistant explains warranty     ->  kept in full
```

**Pros:** Retains key context from the entire conversation. Bounded growth.
**Cons:** Summarization loses detail. Adds an extra LLM call. Summary quality varies.

---

## Context Failure Modes

Understanding how context can fail is just as important as knowing how to build it.

### Context Rot

As the conversation grows, the system prompt gets pushed further from the model's attention. Instructions that were followed perfectly in Turn 1 start being ignored by Turn 15. The model's attention to the system prompt "rots" as more content is added between it and the current generation point.

**Mitigation:** Repeat key instructions, use summary memory to keep context compact, or re-inject critical constraints near the end of the prompt.

### Lost Instructions

When the context window fills up, something has to give. If you prioritize conversation history and retrieved documents, the system prompt may get truncated or compressed. The model then loses its behavioral guidelines.

**Mitigation:** Always reserve a fixed token budget for the system prompt. Trim history and retrieval before touching the system prompt.

### Token Overflow

When the total context exceeds the model's maximum window size, the API will either truncate from the beginning (losing the system prompt) or return an error. Either outcome breaks the application.

**Mitigation:** Track token counts for every component. Implement hard limits and pruning strategies. Monitor usage at runtime.

---

## Design Patterns

### Planning

Before generating a response, have the model outline what it needs to do. This can be done in the system prompt ("First, identify the relevant information. Then, formulate your response.") or as an explicit chain-of-thought step.

### Reflection

After generating a draft response, ask the model to review it against the constraints. Does it match the output format? Does it reference only the provided context? This can be done as a second pass or built into the system prompt.

### Active Context Management

Do not treat the context window as a passive container. Actively manage it:
- Monitor token usage per section after every turn
- Summarize history when it exceeds a threshold
- Re-rank or filter retrieved documents by relevance score
- Drop low-relevance documents rather than including everything
- Reserve guaranteed token budgets for each section

---

## Practical Tips

### When to Summarize
- When conversation history exceeds 40-50% of your token budget
- When the user shifts to a new topic (the old context is less relevant)
- At natural breakpoints (e.g., after a decision is made)

### When to Prune
- When total context approaches 80% of the window size
- When retrieved documents have low relevance scores
- When conversation turns are repetitive or contain pleasantries

### When to Compress
- System prompts that have grown too large from accumulated instructions
- Retrieved documents that contain boilerplate or formatting overhead
- Chat history that includes verbose assistant responses

### Token Budget Guidelines (for a 4096-token window)
- System Prompt: ~250-400 tokens (6-10%)
- Retrieved Documents: ~1000-1500 tokens (25-35%)
- Conversation History: ~500-1000 tokens (12-25%)
- User Query: ~50-100 tokens (1-3%)
- Available for Response: ~1000-1500 tokens (25-35%)

---

## Scripts in This Phase

### 01_prompt_engineering.py

Demonstrates three levels of system prompt design (basic, structured, XML-tagged) by sending the same query to the LLM with each prompt and comparing the responses.

```bash
python 01_prompt_engineering.py
```

### 02_conversation_memory.py

Implements three conversation memory strategies from scratch (buffer, window, summary) and simulates a multi-turn conversation to show how each handles long-term context.

```bash
python 02_conversation_memory.py
```

### 03_context_window.py

Visualizes context window token usage across all four components and demonstrates what happens when the window fills up, including a pruning strategy.

```bash
python 03_context_window.py
```

### 04_chainlit_app.py

Enhanced Chainlit chatbot that applies all context engineering techniques: XML-tagged system prompts, window-based conversation memory, relevance-filtered retrieval, guardrails, and token usage monitoring.

```bash
chainlit run 04_chainlit_app.py -w
```

---

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Together.ai API key in a `.env` file:
   ```
   TOGETHER_API_KEY=your_key_here
   ```

3. Ensure the `data/` directory contains the dealership data files (inventory.csv, complaints.csv, policies.txt, promotions.md, faq.json).
