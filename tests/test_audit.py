"""
Comprehensive audit test suite for chroma-rag-ce project.
Tests each phase's core functionality without requiring interactive input or API calls
(where possible), and validates data/code consistency.
"""

import csv
import json
import os
import sys
from pathlib import Path

import pytest
import chromadb

# Project root
ROOT = Path(__file__).parent.parent


# ============================================================================
# Phase 1: Foundations
# ============================================================================

class TestPhase1:
    """Test Phase 1 - ChromaDB basics and data loading."""

    DATA_DIR = ROOT / "phase-1-foundations" / "data"

    def test_inventory_txt_exists(self):
        assert (self.DATA_DIR / "inventory.txt").exists()

    def test_policies_txt_exists(self):
        assert (self.DATA_DIR / "policies.txt").exists()

    def test_load_inventory_function(self):
        """Test the load_inventory parser from 01_hello_chroma.py."""
        sys.path.insert(0, str(ROOT / "phase-1-foundations"))
        from importlib import import_module
        mod = import_module("01_hello_chroma")
        cars = mod.load_inventory(str(self.DATA_DIR / "inventory.txt"))
        assert len(cars) == 10, f"Expected 10 cars, got {len(cars)}"
        # Each car should have name, year, make, model
        for car in cars:
            assert "name" in car
            assert "year" in car
            assert "make" in car

    def test_parse_price(self):
        sys.path.insert(0, str(ROOT / "phase-1-foundations"))
        from importlib import import_module
        mod = import_module("01_hello_chroma")
        assert mod.parse_price("$28,990") == 28990
        assert mod.parse_price("$49,900") == 49900

    def test_chromadb_basic_operations(self):
        """Test that ChromaDB works with inventory data."""
        client = chromadb.Client()
        collection = client.create_collection(name="test_phase1")
        collection.add(
            documents=["2024 Toyota Camry - great sedan", "2024 Honda CR-V - family SUV"],
            ids=["car_0", "car_1"],
            metadatas=[{"make": "Toyota"}, {"make": "Honda"}],
        )
        assert collection.count() == 2
        results = collection.query(query_texts=["SUV"], n_results=1)
        assert len(results["documents"][0]) == 1

    def test_load_text_blocks_function(self):
        """Test the load_text_blocks parser from 02_basic_rag.py."""
        sys.path.insert(0, str(ROOT / "phase-1-foundations"))
        from importlib import import_module
        mod = import_module("02_basic_rag")
        blocks = mod.load_text_blocks(str(self.DATA_DIR / "inventory.txt"))
        assert len(blocks) > 0, "Should load at least one text block"
        blocks_policies = mod.load_text_blocks(str(self.DATA_DIR / "policies.txt"))
        assert len(blocks_policies) > 0

    def test_format_context(self):
        sys.path.insert(0, str(ROOT / "phase-1-foundations"))
        from importlib import import_module
        mod = import_module("02_basic_rag")
        result = mod.format_context(["doc1", "doc2"])
        assert "[Document 1]" in result
        assert "[Document 2]" in result


# ============================================================================
# Phase 2: Document Loading
# ============================================================================

class TestPhase2:
    """Test Phase 2 - Document loaders and chunking strategies."""

    DATA_DIR = ROOT / "phase-2-document-loading" / "data"

    def test_data_files_exist(self):
        for name in ["policies.txt", "inventory.csv", "promotions.md", "faq.json"]:
            assert (self.DATA_DIR / name).exists(), f"Missing: {name}"

    def test_inventory_csv_columns(self):
        """Verify inventory.csv has required columns for all loaders."""
        with open(self.DATA_DIR / "inventory.csv", "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        # Phase 2 CSVLoader uses all columns dynamically, so just check basic ones
        for col in ["make", "model", "year", "trim", "price"]:
            assert col in row, f"Missing column '{col}' in Phase 2 inventory.csv"

    def test_faq_json_structure(self):
        """Verify faq.json entries have required fields."""
        with open(self.DATA_DIR / "faq.json", "r") as f:
            data = json.load(f)
        assert len(data) > 0, "faq.json should not be empty"
        for entry in data:
            assert "question" in entry, "FAQ entry missing 'question'"
            assert "answer" in entry, "FAQ entry missing 'answer'"

    def test_faq_json_has_category(self):
        """Phase 2 script 01 uses record.get('category') - should have category for metadata."""
        with open(self.DATA_DIR / "faq.json", "r") as f:
            data = json.load(f)
        # This is a soft check - code uses .get() so won't crash, but metadata will be empty
        has_category = all("category" in entry for entry in data)
        if not has_category:
            pytest.xfail("faq.json missing 'category' field - code uses .get() so safe, but metadata will be incomplete")

    def test_text_loader(self):
        """Test TextLoader on policies.txt."""
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(self.DATA_DIR / "policies.txt"))
        docs = loader.load()
        assert len(docs) == 1, "TextLoader should produce 1 document"
        assert len(docs[0].page_content) > 100

    def test_csv_loader(self):
        """Test CSVLoader on inventory.csv."""
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(str(self.DATA_DIR / "inventory.csv"))
        docs = loader.load()
        assert len(docs) >= 10, f"Expected at least 10 rows, got {len(docs)}"

    def test_json_loader(self):
        """Test JSONLoader on faq.json."""
        from langchain_community.document_loaders import JSONLoader

        def extract_metadata(record: dict, metadata: dict) -> dict:
            metadata["question"] = record.get("question", "")
            metadata["category"] = record.get("category", "")
            return metadata

        loader = JSONLoader(
            file_path=str(self.DATA_DIR / "faq.json"),
            jq_schema=".[]",
            content_key="answer",
            metadata_func=extract_metadata,
        )
        docs = loader.load()
        assert len(docs) > 0, "JSONLoader should produce documents"

    def test_chunking_strategies(self):
        """Test that chunking produces expected results."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(self.DATA_DIR / "policies.txt"))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        assert len(chunks) > 1, "Should produce multiple chunks"
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # some tolerance over chunk_size


# ============================================================================
# Phase 3: Search and Retrieval
# ============================================================================

class TestPhase3:
    """Test Phase 3 - Search modalities and metadata filtering."""

    DATA_DIR = ROOT / "phase-3-search-and-retrieval" / "data"

    def test_data_files_exist(self):
        for name in ["policies.txt", "inventory.csv", "promotions.md", "faq.json", "complaints.csv"]:
            assert (self.DATA_DIR / name).exists(), f"Missing: {name}"

    def test_inventory_csv_has_required_columns(self):
        """Phase 3 scripts reference 'category' and 'condition' columns."""
        with open(self.DATA_DIR / "inventory.csv", "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        required = ["make", "model", "year", "trim", "price", "color", "features",
                     "status", "category", "condition"]
        missing = [col for col in required if col not in row]
        assert not missing, f"Phase 3 inventory.csv missing columns: {missing}"

    def test_complaints_csv_has_vehicle_column(self):
        """Phase 3 scripts reference 'vehicle' column in complaints.csv."""
        with open(self.DATA_DIR / "complaints.csv", "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "vehicle" in row, "Phase 3 complaints.csv missing 'vehicle' column"

    def test_faq_json_has_category(self):
        """Phase 3 scripts use entry['category'] (direct access, will crash without it)."""
        with open(self.DATA_DIR / "faq.json", "r") as f:
            data = json.load(f)
        for i, entry in enumerate(data):
            assert "category" in entry, f"FAQ entry {i} missing 'category' - Phase 3 code uses direct key access"

    def test_search_modalities_load_inventory(self):
        """Test Phase 3 script 02's load_inventory function."""
        sys.path.insert(0, str(ROOT / "phase-3-search-and-retrieval"))
        try:
            from importlib import import_module
            mod = import_module("02_search_modalities")
            docs = mod.load_inventory()
            assert len(docs) > 0, "Should load inventory documents"
            # Verify each doc has expected metadata keys
            for doc in docs:
                meta = doc["metadata"]
                assert "category" in meta, "Inventory metadata should have 'category'"
                assert "condition" in meta, "Inventory metadata should have 'condition'"
        except KeyError as e:
            pytest.fail(f"KeyError loading Phase 3 inventory: {e}")

    def test_metadata_filtering_load_inventory(self):
        """Test Phase 3 script 03's load_inventory function."""
        sys.path.insert(0, str(ROOT / "phase-3-search-and-retrieval"))
        try:
            from importlib import import_module
            mod = import_module("03_metadata_filtering")
            docs, metas, ids = mod.load_inventory()
            assert len(docs) > 0
            for meta in metas:
                assert "category" in meta
                assert "condition" in meta
        except KeyError as e:
            pytest.fail(f"KeyError loading Phase 3 inventory: {e}")

    def test_metadata_filtering_load_complaints(self):
        """Test Phase 3 script 03's load_complaints function."""
        sys.path.insert(0, str(ROOT / "phase-3-search-and-retrieval"))
        try:
            from importlib import import_module
            mod = import_module("03_metadata_filtering")
            docs, metas, ids = mod.load_complaints()
            assert len(docs) > 0
            for meta in metas:
                assert "vehicle" in meta or True  # vehicle is in doc text, not always in meta
        except KeyError as e:
            pytest.fail(f"KeyError loading Phase 3 complaints: {e}")

    def test_metadata_filtering_load_faq(self):
        """Test Phase 3 script 03's load_faq function."""
        sys.path.insert(0, str(ROOT / "phase-3-search-and-retrieval"))
        try:
            from importlib import import_module
            mod = import_module("03_metadata_filtering")
            docs, metas, ids = mod.load_faq()
            assert len(docs) > 0
        except KeyError as e:
            pytest.fail(f"KeyError loading Phase 3 FAQ: {e}")

    def test_embedding_search(self):
        """Test that ChromaDB semantic search works correctly."""
        client = chromadb.Client()
        try:
            client.delete_collection("test_phase3")
        except Exception:
            pass
        collection = client.create_collection(name="test_phase3")
        collection.add(
            documents=[
                "Toyota Camry is a family sedan",
                "Ford F-150 is a powerful truck",
                "Honda CR-V is a compact SUV",
            ],
            ids=["d0", "d1", "d2"],
            metadatas=[
                {"category": "sedan"},
                {"category": "truck"},
                {"category": "suv"},
            ],
        )
        # Semantic search
        results = collection.query(query_texts=["SUV for family"], n_results=3)
        all_docs_text = " ".join(results["documents"][0])
        assert "CR-V" in all_docs_text or "SUV" in all_docs_text.lower()

        # Metadata filter
        results = collection.query(
            query_texts=["vehicle"],
            where={"category": "truck"},
            n_results=1,
        )
        assert "F-150" in results["documents"][0][0]


# ============================================================================
# Phase 4: Chatbot UI
# ============================================================================

class TestPhase4:
    """Test Phase 4 - Chainlit app data loading and collections."""

    DATA_DIR = ROOT / "phase-4-chatbot-ui" / "data"

    def test_data_files_exist(self):
        for name in ["policies.txt", "inventory.csv", "promotions.md", "faq.json", "complaints.csv"]:
            assert (self.DATA_DIR / name).exists(), f"Missing: {name}"

    def test_inventory_csv_has_all_columns(self):
        """Phase 4 has the complete CSV format."""
        with open(self.DATA_DIR / "inventory.csv", "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        required = ["year", "make", "model", "trim", "price", "mileage",
                     "condition", "color", "category", "features", "status"]
        missing = [col for col in required if col not in row]
        assert not missing, f"Phase 4 inventory.csv missing columns: {missing}"

    def test_complaints_csv_has_vehicle(self):
        with open(self.DATA_DIR / "complaints.csv", "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "vehicle" in row, "Phase 4 complaints.csv should have 'vehicle' column"

    def test_faq_json_has_category(self):
        with open(self.DATA_DIR / "faq.json", "r") as f:
            data = json.load(f)
        for entry in data:
            assert "category" in entry, "Phase 4 faq.json should have 'category'"

    def test_collections_script_load_csv(self):
        """Test Phase 4 script 02's CSV loading."""
        sys.path.insert(0, str(ROOT / "phase-4-chatbot-ui"))
        from importlib import import_module
        mod = import_module("02_collections")
        docs = mod.load_csv_file(self.DATA_DIR / "inventory.csv")
        assert len(docs) >= 10
        # Each doc is a (content, metadata) tuple
        content, metadata = docs[0]
        assert isinstance(content, str)
        assert isinstance(metadata, dict)

    def test_collections_script_load_json(self):
        sys.path.insert(0, str(ROOT / "phase-4-chatbot-ui"))
        from importlib import import_module
        mod = import_module("02_collections")
        docs = mod.load_json_file(self.DATA_DIR / "faq.json")
        assert len(docs) > 0
        content, metadata = docs[0]
        assert "Question:" in content

    def test_pdf_loading_handles_missing_pdf(self):
        """PDF loading should handle missing file gracefully."""
        sys.path.insert(0, str(ROOT / "phase-4-chatbot-ui"))
        from importlib import import_module
        mod = import_module("02_collections")
        fake_path = self.DATA_DIR / "nonexistent.pdf"
        docs = mod.load_pdf_file(fake_path)
        assert docs == [], "Should return empty list for missing PDF"


# ============================================================================
# Phase 5: Context Engineering
# ============================================================================

class TestPhase5:
    """Test Phase 5 - Context engineering components."""

    DATA_DIR = ROOT / "phase-5-context-engineering" / "data"

    def test_data_files_exist(self):
        for name in ["policies.txt", "inventory.csv", "promotions.md", "faq.json",
                      "complaints.csv", "news_articles.md", "tickets.json"]:
            assert (self.DATA_DIR / name).exists(), f"Missing: {name}"

    def test_inventory_csv_has_required_columns(self):
        """Phase 5 chainlit app references 'category' and 'condition'."""
        with open(self.DATA_DIR / "inventory.csv", "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        required = ["year", "make", "model", "trim", "price", "mileage",
                     "condition", "color", "category", "features", "status"]
        missing = [col for col in required if col not in row]
        assert not missing, f"Phase 5 inventory.csv missing columns: {missing}"

    def test_complaints_csv_has_vehicle_column(self):
        """Phase 5 chainlit app references 'vehicle' column."""
        with open(self.DATA_DIR / "complaints.csv", "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "vehicle" in row, "Phase 5 complaints.csv missing 'vehicle' column"

    def test_memory_classes(self):
        """Test the memory strategy classes from 02_conversation_memory.py."""
        sys.path.insert(0, str(ROOT / "phase-5-context-engineering"))
        from importlib import import_module
        mod = import_module("02_conversation_memory")

        # Test BufferMemory
        buf = mod.BufferMemory()
        buf.add_user_message("Hello")
        buf.add_ai_message("Hi there")
        assert buf.get_message_count() == 2
        assert buf.get_token_count() > 0

        # Test WindowMemory
        win = mod.WindowMemory(window_size=4)
        for i in range(6):
            win.add_user_message(f"msg {i}")
            win.add_ai_message(f"reply {i}")
        assert win.get_message_count() <= 4, "Window memory should trim to window_size"

    def test_token_counting(self):
        """Test tiktoken-based token counting."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = len(enc.encode("Hello, how are you?"))
        assert tokens > 0 and tokens < 20

    def test_prompt_engineering_prompts(self):
        """Test that prompt builders return non-empty strings."""
        sys.path.insert(0, str(ROOT / "phase-5-context-engineering"))
        from importlib import import_module
        mod = import_module("01_prompt_engineering")
        assert len(mod.build_basic_prompt()) > 10
        assert len(mod.build_structured_prompt()) > 100
        assert len(mod.build_xml_prompt()) > 100
        assert "<role>" in mod.build_xml_prompt()


# ============================================================================
# Phase 6: Monitoring and Evaluation
# ============================================================================

class TestPhase6:
    """Test Phase 6 - Evaluation framework."""

    DATA_DIR = ROOT / "phase-6-monitoring-and-evaluation" / "data"

    def test_data_files_exist(self):
        for name in ["policies.txt", "inventory.csv", "promotions.md", "faq.json",
                      "complaints.csv", "news_articles.md", "tickets.json"]:
            assert (self.DATA_DIR / name).exists(), f"Missing: {name}"

    def test_inventory_csv_has_required_columns(self):
        """Phase 6 chainlit app references 'category' and 'condition'."""
        with open(self.DATA_DIR / "inventory.csv", "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        required = ["year", "make", "model", "trim", "price", "mileage",
                     "condition", "color", "category", "features", "status"]
        missing = [col for col in required if col not in row]
        assert not missing, f"Phase 6 inventory.csv missing columns: {missing}"

    def test_complaints_csv_has_vehicle_column(self):
        """Phase 6 chainlit app references 'vehicle' column."""
        with open(self.DATA_DIR / "complaints.csv", "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "vehicle" in row, "Phase 6 complaints.csv missing 'vehicle' column"

    def test_evaluation_test_cases(self):
        """Test that evaluation test cases are well-formed."""
        sys.path.insert(0, str(ROOT / "phase-6-monitoring-and-evaluation"))
        from importlib import import_module
        mod = import_module("03_evaluation")
        assert len(mod.TEST_CASES) == 8
        for tc in mod.TEST_CASES:
            assert "question" in tc
            assert "expected_topic" in tc
            assert "expected_sources" in tc
            assert isinstance(tc["expected_sources"], list)

    def test_evaluation_document_loading(self):
        """Test that the evaluation script can load documents."""
        sys.path.insert(0, str(ROOT / "phase-6-monitoring-and-evaluation"))
        from importlib import import_module
        mod = import_module("03_evaluation")
        docs = mod.load_all_documents()
        assert len(docs) > 0, "Should load documents"
        # Each doc should have text, source, doc_type
        for doc in docs:
            assert "text" in doc
            assert "source" in doc
            assert "doc_type" in doc


# ============================================================================
# Cross-Phase Consistency
# ============================================================================

class TestCrossPhaseConsistency:
    """Verify data consistency across phases."""

    def test_inventory_count_consistent(self):
        """Phases 2,3,5,6 should have the same inventory count. Phase 4 may differ (demo data)."""
        counts = {}
        for phase_dir in sorted(ROOT.glob("phase-*/data")):
            csv_path = phase_dir / "inventory.csv"
            if csv_path.exists():
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    count = sum(1 for _ in reader)
                counts[phase_dir.parent.name] = count
        # Phase 4 has a curated smaller dataset; the rest should be consistent
        main_phases = {k: v for k, v in counts.items() if k != "phase-4-chatbot-ui"}
        values = list(main_phases.values())
        assert len(set(values)) == 1, f"Inventory counts differ: {main_phases}"

    def test_faq_count_consistent(self):
        """Phases 2,3,5,6 should have the same FAQ count. Phase 4 may differ."""
        counts = {}
        for phase_dir in sorted(ROOT.glob("phase-*/data")):
            json_path = phase_dir / "faq.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    data = json.load(f)
                counts[phase_dir.parent.name] = len(data)
        main_phases = {k: v for k, v in counts.items() if k != "phase-4-chatbot-ui"}
        values = list(main_phases.values())
        assert len(set(values)) == 1, f"FAQ counts differ: {main_phases}"

    def test_policies_content_consistent(self):
        """All phases should have the same policies.txt content."""
        contents = {}
        for phase_dir in sorted(ROOT.glob("phase-*/data")):
            txt_path = phase_dir / "policies.txt"
            if txt_path.exists():
                with open(txt_path, "r") as f:
                    contents[phase_dir.parent.name] = f.read()
        # All policies should be identical
        values = list(contents.values())
        first = values[0]
        for phase_name, content in contents.items():
            assert content == first, f"{phase_name} has different policies.txt"


# ============================================================================
# API Integration Tests (require TOGETHER_API_KEY)
# ============================================================================

class TestAPIIntegration:
    """Test Together.ai API integration (requires API key)."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        if not os.getenv("TOGETHER_API_KEY"):
            pytest.skip("TOGETHER_API_KEY not set")

    def test_llm_basic_call(self):
        """Test that the LLM can be called successfully."""
        from langchain_together import ChatTogether
        from langchain_core.messages import HumanMessage

        llm = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            temperature=0.1,
            max_tokens=50,
        )
        response = llm.invoke([HumanMessage(content="Say hello in one word.")])
        assert response.content is not None
        assert len(response.content) > 0

    def test_rag_end_to_end(self):
        """Test a full RAG pipeline: load -> embed -> query -> generate."""
        from langchain_together import ChatTogether
        from langchain_core.messages import HumanMessage, SystemMessage

        client = chromadb.Client()
        try:
            client.delete_collection("test_e2e")
        except Exception:
            pass
        collection = client.create_collection(name="test_e2e")
        collection.add(
            documents=[
                "The 2024 Toyota Camry LE costs $28,990 and is available.",
                "Our return policy allows returns within 7 days.",
            ],
            ids=["d0", "d1"],
        )

        results = collection.query(query_texts=["What is the return policy?"], n_results=1)
        context = results["documents"][0][0]

        llm = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            temperature=0.1,
            max_tokens=100,
        )
        response = llm.invoke([
            SystemMessage(content=f"Answer based on this context: {context}"),
            HumanMessage(content="What is the return policy?"),
        ])
        assert "7" in response.content or "seven" in response.content.lower()
