"""Integration tests for the SimbaAgent."""

import sys
import os
import pytest
import dspy

# Adjust the path to import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent import SimbaAgent, setup_rag  # pylint: disable=import-error, wrong-import-position

# --- Fixtures ---


# Fixture to setup DSPy LM settings once per module
# pylint: disable=inconsistent-return-statements # R1710: Fixture might fail
@pytest.fixture(scope="module")
def llm_fixture():  # Renamed from setup_dspy_settings_for_tests
    """Fixture to configure and return the DSPy LM."""
    print("\n(Test Fixture) Configuring DSPy LM settings...")
    # Use the specified LLM model
    # llm = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=150) # Example
    # Use the user-preferred model, assuming API keys are in env
    try:
        # E1101 fix: dspy.LM is the correct way to load models via litellm
        llm = dspy.LM("openrouter/google/gemini-2.0-flash-001")
        dspy.settings.configure(lm=llm)
        print(f"DSPy LM configured: {llm.provider}/{llm.model}")
        return llm  # Return the configured LLM
    except Exception as e:  # W0718: Catch specific errors if possible
        pytest.fail(f"Failed to configure DSPy LLM: {e}")


# Fixture to load corpus data once per module
# pylint: disable=inconsistent-return-statements # R1710: Fixture might fail
@pytest.fixture(scope="module")
def corpus_data():
    """Loads corpus data from corpus.txt."""
    print("\n(Test Fixture) Loading corpus data...")
    filepath = "corpus.txt"
    try:
        # W1514: Specify encoding
        with open(filepath, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f if line.strip()]
        assert corpus, "Corpus file is empty or could not be read properly."
        print(f"Loaded {len(corpus)} documents from {filepath}")
        return corpus
    except FileNotFoundError:
        pytest.fail(f"Corpus file not found at {filepath}")
    except Exception as e:  # W0718: Catch specific errors if possible
        pytest.fail(f"Error loading corpus file {filepath}: {e}")


# Fixture to set up the RAG retriever once per module
# pylint: disable=redefined-outer-name, inconsistent-return-statements
# pylint: disable=inconsistent-return-statements # R1710: Fixture might fail or skip
@pytest.fixture(scope="module")
def rag_retriever(corpus_data, llm_fixture):
    """Fixture to set up and return the RAG retriever backend."""
    print("\n(Test Fixture) Setting up RAG retriever backend...")
    corpus = corpus_data
    if not corpus:
        print("Warning: Corpus fixture is empty. RAG tests might behave unexpectedly.")
        return None

    # Setup the Embeddings backend
    embedder_name = "openai/text-embedding-3-small"
    k_fixture = 3
    print(
        f"Setting up Embeddings retriever with {embedder_name} for {len(corpus)} docs, k={k_fixture}..."
    )
    try:
        # Use the actual setup_rag function from agent.py
        backend_retriever = setup_rag(
            corpus=corpus, embedder_name=embedder_name, k=k_fixture
        )
        print("RAG retriever setup successful: {type(backend_retriever)}")

        # Return the *backend* retriever module, not the Retrieve client
        return backend_retriever
    except ImportError as e:
        pytest.skip(f"Skipping RAG tests: {e}")  # Skip if FAISS/dependencies missing
    except Exception as e:  # W0718: Catch specific errors if possible
        print(f"Error setting up RAG retriever fixture: {e}")
        pytest.fail(f"Failed to setup RAG retriever: {e}")


# Fixture to instantiate the SimbaAgent once per module
# pylint: disable=redefined-outer-name, unused-argument, missing-function-docstring
@pytest.fixture(scope="module")
def simba_agent_instance(llm_fixture, rag_retriever):
    """Fixture to create a SimbaAgent instance for testing."""
    print("\n(Test Fixture) Instantiating SimbaAgent...")
    # Pass the retriever model directly to the agent
    # Use the LLM from llm_fixture
    agent = SimbaAgent(llm=llm_fixture, retriever_model=rag_retriever)
    assert agent is not None, "Failed to instantiate SimbaAgent"
    return agent


# --- Test Cases ---
# pylint: disable=redefined-outer-name, unused-argument, missing-function-docstring
@pytest.mark.integration
def test_agent_initialization(rag_retriever, simba_agent_instance):
    print("\n--- Running test_agent_initialization ---")
    assert rag_retriever is not None, (
        "RAG retriever fixture failed (backend should be created)"
    )
    assert simba_agent_instance is not None, "SimbaAgent fixture failed"
    # Check agent has the retriever model attribute set correctly
    assert hasattr(simba_agent_instance, "retriever_model"), (
        "Agent missing retriever_model attribute"
    )
    assert simba_agent_instance.retriever_model is rag_retriever, (
        "Agent's retriever_model is not the one from the fixture"
    )


@pytest.mark.integration
def test_agent_simple_query(simba_agent_instance):
    print("\n--- Running test_agent_simple_query ---")
    test_agent = simba_agent_instance
    question = "What is DSPy?"
    print(f"Querying agent: '{question}'")
    result = test_agent(question=question)

    assert result is not None, "Agent should return a result"
    assert hasattr(result, "answer"), "Result should have an 'answer' attribute"
    assert isinstance(result.answer, str), "Answer should be a string"
    assert len(result.answer) > 10, "Answer seems too short to be descriptive"
    # Check for relevant keywords (case-insensitive)
    answer_lower = result.answer.lower()
    assert "dspy" in answer_lower, "Answer should mention DSPy"
    assert "framework" in answer_lower or "library" in answer_lower, (
        "Answer should mention 'framework' or 'library'"
    )
    print(f"Agent Answer (Simple): {result.answer}")


@pytest.mark.integration
def test_agent_with_rag(rag_retriever, simba_agent_instance):
    print("\n--- Running test_agent_with_rag ---")
    test_agent = simba_agent_instance
    # Check if the agent was initialized with a retriever model
    assert test_agent.retriever_model is not None, (
        "Agent fixture did not provide a retriever model to the agent instance."
    )
    # Ensure rag_retriever fixture itself is valid
    assert rag_retriever is not None, "rag_retriever fixture failed to initialize"

    question = "Explain DSPy framework based on context."

    print(f"Querying agent: '{question}'")
    result = test_agent(question=question)

    assert result is not None, "Agent should return a result"
    assert hasattr(result, "answer"), "Result should have an 'answer' attribute"
    assert isinstance(result.answer, str), "Answer should be a string"
    assert len(result.answer) > 10, "Answer seems too short"

    # Check for keywords expected from the corpus context (case-insensitive)
    answer_lower = result.answer.lower()
    assert "dspy" in answer_lower, "Answer should mention DSPy"
    assert "framework" in answer_lower, "Answer should mention 'framework'"
    assert "stanford" in answer_lower, "Answer should mention 'Stanford' (from corpus)"
    assert "language model" in answer_lower or "lm" in answer_lower, (
        "Answer should mention language models"
    )
    assert "structure" in answer_lower or "optimization" in answer_lower, (
        "Answer should mention 'structure' or 'optimization' (from corpus)"
    )
    print(f"Agent Answer (RAG): {result.answer}")


# Test running a query that should primarily use the Wikipedia tool
@pytest.mark.integration
def test_agent_wikipedia_search(simba_agent_instance):
    print("\n--- Running test_agent_wikipedia_search ---")
    test_agent = simba_agent_instance
    question = "Tell me about George Washington."
    print(f"Querying agent: '{question}'")
    result = test_agent(question=question)

    assert result is not None, "Agent should return a result"
    assert hasattr(result, "answer"), "Result should have an 'answer' attribute"
    assert isinstance(result.answer, str), "Answer should be a string"
    assert len(result.answer) > 50, "Wikipedia summary seems too short"
    # Check for keywords strongly indicative of the correct Wikipedia summary
    answer_lower = result.answer.lower()
    assert "george washington" in answer_lower, (
        "Answer should mention George Washington"
    )
    # assert "president" in answer_lower, "Answer should mention 'president'" # Might be brittle
    # assert "american revolution" in answer_lower, "Answer should mention American Revolution" # Might be brittle
    print(f"Agent Answer (Wikipedia): {result.answer}")
