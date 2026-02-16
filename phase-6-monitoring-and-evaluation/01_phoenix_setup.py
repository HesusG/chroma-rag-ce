"""
Phase 6, Script 1: Setting Up Arize Phoenix

This script demonstrates how to set up Arize Phoenix for LLM observability.
Phoenix provides tracing, evaluation, and monitoring for LLM applications.

What this script does:
  1. Launches Phoenix as a background server (dashboard at localhost:6006)
  2. Configures OpenTelemetry to send traces to Phoenix
  3. Instruments LangChain so all LLM calls are automatically traced
  4. Runs a single LLM call so you can see a trace in the dashboard

Key concepts:
  - Phoenix is an open-source LLM observability tool by Arize AI
  - OpenTelemetry (OTel) is the standard protocol for distributed tracing
  - OpenInference is the instrumentation layer that auto-traces LangChain
  - A "trace" is a complete record of one operation (e.g., one LLM call)
  - A "span" is a single step within a trace (e.g., the LLM invocation)

After running this script, open http://localhost:6006 to see the Phoenix dashboard.

Requirements:
  - TOGETHER_API_KEY set in .env file or as an environment variable
  - arize-phoenix, openinference-instrumentation-langchain, and related packages

Usage:
  python 01_phoenix_setup.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def setup_phoenix():
    """
    Initialize Phoenix tracing.

    This function does three things:
      1. Launches Phoenix as a local server (http://localhost:6006)
      2. Registers an OpenTelemetry tracer provider that sends spans to Phoenix
      3. Instruments LangChain so all LLM/chain/retriever calls are auto-traced

    After calling this function, any LangChain operation in your application
    will automatically generate traces visible in the Phoenix dashboard.

    Returns:
        The Phoenix session object (contains the dashboard URL).
    """
    import phoenix as px

    # Launch Phoenix in the background. This starts a local web server
    # at http://localhost:6006 that collects and displays traces.
    session = px.launch_app()
    print(f"  Phoenix dashboard: {session.url}")

    # Set up OpenTelemetry tracing. The register() function creates a
    # tracer provider configured to export spans to the local Phoenix instance.
    # The project_name groups all traces under a single project in the dashboard.
    from phoenix.otel import register

    tracer_provider = register(project_name="autoelite-rag")
    print("  OpenTelemetry tracer provider registered.")

    # Instrument LangChain. This monkey-patches LangChain's internal classes
    # so that every LLM call, chain execution, and retriever query automatically
    # creates OpenTelemetry spans. You do not need to modify your LangChain code.
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    print("  LangChain instrumentation active.")

    return session


def run_traced_llm_call():
    """
    Run a simple LLM call that will be traced by Phoenix.

    This is a basic LangChain ChatTogether call with a system message and a
    human message. Because we instrumented LangChain in setup_phoenix(), this
    call automatically generates a trace with:
      - The full prompt (system + human messages)
      - The model's response
      - Token counts (input, output, total)
      - Latency (time from request to response)
      - Model name and provider

    Open the Phoenix dashboard to see all of this information.
    """
    from langchain_together import ChatTogether
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatTogether(
        model=os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        temperature=0.3,
        max_tokens=512,
    )

    messages = [
        SystemMessage(
            content="You are a helpful car dealership assistant for AutoElite Motors. "
            "You help customers find the right vehicle and answer questions about "
            "inventory, financing, and dealership policies."
        ),
        HumanMessage(content="What should I consider when buying my first car?"),
    ]

    print("\n  Sending a traced LLM call...")
    response = llm.invoke(messages)
    print(f"\n  Response (first 500 chars):\n  {response.content[:500]}")

    # Display token usage if available in the response metadata
    if hasattr(response, "response_metadata") and response.response_metadata:
        metadata = response.response_metadata
        usage = metadata.get("token_usage", metadata.get("usage", {}))
        if usage:
            print(f"\n  Token usage:")
            print(f"    Input tokens:  {usage.get('prompt_tokens', 'N/A')}")
            print(f"    Output tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"    Total tokens:  {usage.get('total_tokens', 'N/A')}")

    return response


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6: Setting Up Arize Phoenix")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print()
        print("  ERROR: TOGETHER_API_KEY not found.")
        print("  Set it in a .env file or export it in your terminal:")
        print()
        print('    export TOGETHER_API_KEY="your-key-here"')
        print()
        print("  Get a free key at https://api.together.ai/")
        sys.exit(1)

    # Step 1: Launch Phoenix
    print("\n[Step 1] Launching Phoenix...")
    session = setup_phoenix()

    # Step 2: Run a traced LLM call
    print("\n[Step 2] Running a traced LLM call...")
    run_traced_llm_call()

    # Step 3: Prompt user to check the dashboard
    print("\n" + "=" * 60)
    print(f"  Open the Phoenix dashboard at: {session.url}")
    print("  You should see the traced LLM call in the dashboard.")
    print("  Click on the trace to see:")
    print("    - The full prompt (system + human messages)")
    print("    - The model's response")
    print("    - Token counts and latency")
    print("=" * 60)

    # Keep the script running so the Phoenix dashboard stays available.
    # Phoenix runs as an in-process server; when the script exits, the
    # dashboard goes down.
    try:
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        pass

    print("\nPhoenix server stopped. Goodbye.")
