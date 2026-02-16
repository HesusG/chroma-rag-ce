"""
Phase 5 - Script 01: Prompt Engineering
========================================
Demonstrates three levels of system prompt design and compares
how the same query produces different results with each approach.

The three levels:
  1. Basic:      A single sentence with minimal guidance.
  2. Structured: Role, guidelines, and format spelled out in plain text.
  3. XML-Tagged: Same information organized with explicit XML delimiters.

Run:
    python 01_prompt_engineering.py
"""

import os
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Define the three system prompt levels
# ---------------------------------------------------------------------------

def build_basic_prompt():
    """Level 1: Minimal guidance."""
    return "You are a helpful car dealership assistant for AutoElite Motors."


def build_structured_prompt():
    """Level 2: Role, guidelines, and output format in plain text."""
    return """You are a senior sales assistant at AutoElite Motors car dealership.

Your role is to help customers find the right vehicle and provide accurate information about inventory, pricing, financing, and promotions.

Guidelines:
- Always be professional and friendly
- Only provide information based on the context provided
- If you don't know something, say so honestly
- Suggest test drives when appropriate
- Mention current promotions when relevant

Response format:
- Keep responses concise (2-3 paragraphs max)
- Use bullet points for listing options
- Include specific prices and details when available"""


def build_xml_prompt():
    """Level 3: Same content organized with XML tags for clarity."""
    return """<role>
You are a senior sales assistant at AutoElite Motors car dealership. You have deep knowledge of the inventory, financing options, and current promotions.
</role>

<task>
Help customers find the right vehicle by answering their questions accurately based on the provided context. Guide them through the car-buying process.
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


# ---------------------------------------------------------------------------
# 2. Sample context and query
# ---------------------------------------------------------------------------

SAMPLE_CONTEXT = """Here is the current inventory information:

- 2024 Toyota Camry LE: $28,990, New, Celestial Silver, sedan. Features: Toyota Safety Sense 3.0, 8-inch touchscreen, Apple CarPlay, Android Auto, adaptive cruise control. Status: Available.
- 2024 Honda CR-V EX-L: $38,450, New, Radiant Red Metallic, SUV. Features: 1.5L turbo engine, leather seats, heated front seats, panoramic moonroof, wireless charging, Honda Sensing suite. Status: Available.
- 2024 Hyundai Tucson SEL: $31,750, New, Amazon Gray, SUV. Features: 2.5L engine, 8-inch touchscreen, Hyundai SmartSense, blind-spot monitoring, LED headlamps. Status: Available.
- 2024 Ford Bronco Sport Big Bend: $35,890, New, Area 51 Blue, SUV. Features: 1.5L EcoBoost, SYNC 3, Ford Co-Pilot360, terrain management system. Status: Available.

Current promotions:
- Winter Clearance Event: Up to $3,000 off MSRP on select 2024 models. 0% APR for 60 months on approved credit.
- Trade-In Bonus: $1,500 above KBB value on all trade-ins with new vehicle purchase.
- First Responder & Military Discount: Additional $750 off any vehicle."""

TEST_QUERY = "I'm looking for a family SUV under $35,000. What do you have available?"


# ---------------------------------------------------------------------------
# 3. Run each prompt level and compare
# ---------------------------------------------------------------------------

def run_comparison():
    """Send the same query with each prompt level and display results."""

    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0.3,
        max_tokens=512,
    )

    prompts = [
        ("LEVEL 1: Basic Prompt", build_basic_prompt()),
        ("LEVEL 2: Structured Prompt", build_structured_prompt()),
        ("LEVEL 3: XML-Tagged Prompt", build_xml_prompt()),
    ]

    user_message = f"""Here is the context to use when answering:

{SAMPLE_CONTEXT}

Customer question: {TEST_QUERY}"""

    for label, system_prompt in prompts:
        print("\n" + "=" * 70)
        print(f"  {label}")
        print("=" * 70)

        # Show the full system prompt being sent
        print("\n--- System Prompt ---")
        print(system_prompt)
        print("--- End System Prompt ---\n")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        print("Sending to LLM...")
        response = llm.invoke(messages)

        print("\n--- Response ---")
        print(response.content)
        print("--- End Response ---")

    # Summary
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print("""
What to observe in the outputs above:

1. BASIC PROMPT: The model gives a reasonable answer but may include
   information not in the context, use inconsistent formatting, or
   miss opportunities to mention promotions and next steps.

2. STRUCTURED PROMPT: The model follows the guidelines more closely,
   uses bullet points, and tends to mention promotions. But constraints
   may still be missed in longer conversations.

3. XML-TAGGED PROMPT: The model produces the most structured output.
   It starts with a direct answer, uses bullet points with specific
   details, avoids fabricated information, and ends with a next step.
   The XML tags act as anchors that the model references during generation.

Key takeaway: The more structure you provide in the system prompt,
the more consistent and reliable the model's output becomes. XML tags
are not magic -- they simply make the prompt's organization explicit
and unambiguous for the model to parse.
""")


if __name__ == "__main__":
    run_comparison()
