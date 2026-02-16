"""
Phase 4 - Script 1: PDF Loading with LangChain
================================================

This script demonstrates how to load and chunk PDF documents using LangChain's
PyPDFLoader. PDFs are one of the most common document formats in enterprise
knowledge bases -- training manuals, contracts, product specs, and compliance
documents are almost always in PDF format.

What you will learn:
- How PyPDFLoader extracts text from a PDF, one page at a time
- What metadata is automatically attached to each page
- How to chunk PDF content with RecursiveCharacterTextSplitter
- Why chunking matters for RAG (vector databases work best with small, focused chunks)

Requirements:
- pypdf (installed via requirements.txt)
- A PDF file at data/training_manual.pdf (optional -- the script handles missing files)
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    # ---------------------------------------------------------------------------
    # Step 1: Locate the PDF file
    # ---------------------------------------------------------------------------
    # We look for a training manual PDF in the data/ directory. This file is
    # optional -- if it does not exist, we print a helpful message and exit.
    data_dir = Path(__file__).parent / "data"
    pdf_path = data_dir / "training_manual.pdf"

    print("=" * 60)
    print("Phase 4 - PDF Loading with LangChain")
    print("=" * 60)

    try:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")

        # -------------------------------------------------------------------
        # Step 2: Load the PDF with PyPDFLoader
        # -------------------------------------------------------------------
        # PyPDFLoader reads the PDF and creates one Document object per page.
        # Each Document has:
        #   - page_content: the extracted text from that page
        #   - metadata: a dictionary with "source" (file path) and "page" (page number)
        print(f"\nLoading PDF: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        print(f"Number of pages loaded: {len(pages)}")

        # -------------------------------------------------------------------
        # Step 3: Inspect the first page
        # -------------------------------------------------------------------
        # Let's look at what PyPDFLoader gives us for the first page.
        if pages:
            first_page = pages[0]
            print(f"\n--- First Page ---")
            print(f"Metadata: {first_page.metadata}")
            # Truncate the content to 500 characters for readability
            content_preview = first_page.page_content[:500]
            print(f"Content (first 500 chars):\n{content_preview}")
            if len(first_page.page_content) > 500:
                print(f"... (truncated, full page is {len(first_page.page_content)} characters)")

        # -------------------------------------------------------------------
        # Step 4: Chunk the PDF with RecursiveCharacterTextSplitter
        # -------------------------------------------------------------------
        # A single PDF page can easily contain 500+ words. For RAG, smaller
        # chunks work better because:
        #   - Embeddings capture meaning more precisely on focused text
        #   - The LLM's context window is used more efficiently
        #   - Retrieved results are more relevant to the specific question
        #
        # RecursiveCharacterTextSplitter tries to split on natural boundaries
        # (paragraphs, sentences, words) to keep chunks coherent.
        print(f"\n--- Chunking ---")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Target chunk size in characters
            chunk_overlap=200,  # Overlap between chunks to preserve context at boundaries
        )

        # split_documents works directly on LangChain Document objects
        chunks = splitter.split_documents(pages)

        print(f"Number of chunks: {len(chunks)}")
        print(f"(from {len(pages)} pages)")

        if chunks:
            first_chunk = chunks[0]
            print(f"\n--- First Chunk ---")
            print(f"Metadata: {first_chunk.metadata}")
            print(f"Content ({len(first_chunk.page_content)} chars):")
            print(first_chunk.page_content[:300])
            if len(first_chunk.page_content) > 300:
                print("...")

            # Show chunk size distribution
            sizes = [len(c.page_content) for c in chunks]
            print(f"\n--- Chunk Statistics ---")
            print(f"Total chunks: {len(chunks)}")
            print(f"Smallest chunk: {min(sizes)} characters")
            print(f"Largest chunk: {max(sizes)} characters")
            print(f"Average chunk: {sum(sizes) // len(sizes)} characters")

    except FileNotFoundError:
        print(f"\nPDF file not found at: {pdf_path}")
        print()
        print("This is expected! The training manual PDF is optional.")
        print("To try this script, you can:")
        print("  1. Create a simple PDF using any word processor (Google Docs, Word, etc.)")
        print("     and save it as data/training_manual.pdf")
        print("  2. Download any sample PDF from the internet")
        print("  3. Skip this script and move on to 02_collections.py")
        print()
        print("The Chainlit app (03_chainlit_app.py) will work fine without the PDF --")
        print("it simply skips the training manual if the file is not present.")

    except Exception as e:
        print(f"\nError loading PDF: {e}")
        print("Make sure the pypdf package is installed: pip install pypdf")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
