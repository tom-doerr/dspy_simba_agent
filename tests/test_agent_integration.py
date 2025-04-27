# Integration tests for the SimbaAgent
import pytest
import dspy
import os
from dotenv import load_dotenv

# Import necessary classes and functions from the agent module
from agent import SimbaAgent, setup_rag, load_corpus, search_wikipedia

# Load environment variables (needed for API keys)
load_dotenv()

# Fixture to configure DSPy settings once per module
# Note: Relies on agent.py being implicitly loaded or settings configured elsewhere.
@pytest.fixture(scope="module", autouse=True)
def setup_dspy_settings_for_tests():
    print("\n(Test Setup) Explicitly configuring DSPy LM for tests...")
    try:
        test_llm = dspy.LM("deepseek/deepseek-chat")
        dspy.settings.configure(lm=test_llm)
        print("(Test Setup) DSPy configured with deepseek/deepseek-chat.")
    except Exception as e:
        print(f"(Test Setup) WARNING: Failed to configure DeepSeek LM for tests: {e}")
        print("(Test Setup) Falling back to default OpenAI (if configured globally or via env vars).")
        try:
            if os.getenv("OPENAI_API_KEY"):
                fallback_llm = dspy.OpenAI(model="gpt-3.5-turbo") 
                dspy.settings.configure(lm=fallback_llm)
                print("(Test Setup) DSPy configured with fallback OpenAI GPT-3.5-Turbo.")
            else:
                print("(Test Setup) No DEEPSEEK_API_KEY or OPENAI_API_KEY found. Tests requiring LLM might fail.")
        except Exception as fallback_e:
             print(f"(Test Setup) ERROR: Failed to configure any fallback LM: {fallback_e}")
             pytest.skip("Could not configure DSPy LM for integration tests.", allow_module_level=True)

# Fixture to load the corpus once per module
@pytest.fixture(scope="module")
def corpus_data():
    print("\n(Test Fixture) Loading corpus...")
    corpus_path = "corpus.txt"
    if not os.path.exists(corpus_path):
        print(f"Warning: {corpus_path} not found. Creating dummy corpus for test.")
        with open(corpus_path, "w") as f:
            f.write("This is a test document.\nDSPy is a framework for programming language models.")
    loaded_corpus = load_corpus(corpus_path)
    assert loaded_corpus is not None, "Corpus failed to load"
    return loaded_corpus

# Fixture to set up the RAG retriever once per module
@pytest.fixture(scope="module")
def rag_retriever(corpus_data):
    """Fixture to set up and return the RAG retriever backend."""
    print("\n(Test Fixture) Setting up RAG retriever backend...")
    corpus = corpus_data
    if not corpus:
        print("Warning: Corpus fixture is empty. RAG tests might behave unexpectedly.")
        return None

    # Setup the Embeddings backend
    embedder_name = "openai/text-embedding-3-small"
    dimensions = 512 # Match common OpenAI embedding dims
    k_fixture = 3
    print(f"Setting up Embeddings retriever with {embedder_name} ({dimensions}d) for {len(corpus)} docs, k={k_fixture}...")
    try:
        # Use the actual setup_rag function from agent.py
        backend_retriever = setup_rag(corpus=corpus, embedder_name=embedder_name, dimensions=dimensions, k=k_fixture)
        print("Embeddings Retriever backend configured successfully.")

        # Return the *backend* retriever module, not the Retrieve client
        return backend_retriever
    except ImportError as e:
        pytest.skip(f"Skipping RAG tests: {e}") # Skip if FAISS/dependencies missing
    except Exception as e:
        print(f"Error setting up RAG retriever fixture: {e}")
        pytest.fail(f"Failed to setup RAG retriever: {e}")

# Fixture to instantiate the Wikipedia tool once per module
@pytest.fixture(scope="module")
def wikipedia_tool_instance():
    print("\n(Test Fixture) Instantiating Wikipedia tool...")
    # Remove input_variable, dspy infers it
    return dspy.Tool(name="wikipedia_search",
                    desc="Searches Wikipedia for a given query.",
                    func=search_wikipedia)

# Fixture to instantiate the SimbaAgent once per module
@pytest.fixture(scope="module")
def simba_agent_instance(setup_dspy_settings_for_tests, rag_retriever, wikipedia_tool_instance):
    """Fixture to create a SimbaAgent instance for testing."""
    print("\n(Test Fixture) Instantiating SimbaAgent...")
    # Pass the retriever model directly to the agent
    agent = SimbaAgent(llm=dspy.settings.lm, tool=wikipedia_tool_instance, retriever_model=rag_retriever)
    assert agent is not None, "Failed to instantiate SimbaAgent"
    return agent

# Basic test to check initialization of RAG and Agent
@pytest.mark.integration
def test_agent_initialization(rag_retriever, simba_agent_instance):
    print("\n--- Running test_agent_initialization ---")
    assert rag_retriever is not None, "RAG retriever fixture failed (backend should be created)"
    assert simba_agent_instance is not None, "SimbaAgent fixture failed"
    # Check agent has the retriever model attribute set correctly
    assert hasattr(simba_agent_instance, 'retriever_model'), "Agent missing retriever_model attribute"
    assert simba_agent_instance.retriever_model is rag_retriever, "Agent's retriever_model is not the one from the fixture"

# Test a simple query without expecting specific results yet
# Adds timeout via pytest-timeout default
@pytest.mark.integration
def test_agent_simple_query(simba_agent_instance):
    print("\n--- Running test_agent_simple_query ---")
    test_agent = simba_agent_instance 
    question = "What is DSPy?"
    print(f"Querying agent: '{question}'")
    result = test_agent(question=question)

    assert result is not None, "Agent should return a result"
    assert hasattr(result, 'answer'), "Result should have an 'answer' attribute"
    assert isinstance(result.answer, str), "Answer should be a string"
    assert len(result.answer) > 10, "Answer seems too short to be descriptive"
    # Check for relevant keywords (case-insensitive)
    answer_lower = result.answer.lower()
    assert "dspy" in answer_lower, "Answer should mention DSPy"
    assert "framework" in answer_lower or "library" in answer_lower, "Answer should mention 'framework' or 'library'"
    assert "language model" in answer_lower or "lm" in answer_lower, "Answer should mention language models"
    print(f"Agent Answer: {result.answer}") # Print answer for inspection

# Test running a query that should primarily use RAG
@pytest.mark.integration
def test_agent_with_rag(simba_agent_instance):
    print("\n--- Running test_agent_with_rag ---")
    test_agent = simba_agent_instance 
    # Check if the agent was initialized with a retriever model
    assert test_agent.retriever_model is not None, "Agent fixture did not provide a retriever model to the agent instance."

    question = "Explain DSPy framework based on context."
    print(f"Querying agent: '{question}'")
    # The call itself will use the globally configured RAG via dspy.Retrieve
    result = test_agent(question=question)

    assert result is not None, "Agent should return a result"
    assert hasattr(result, 'answer'), "Result should have an 'answer' attribute"
    assert isinstance(result.answer, str), "Answer should be a string"
    assert len(result.answer) > 10, "Answer seems too short"

    # Check for keywords expected from the corpus context (case-insensitive)
    answer_lower = result.answer.lower()
    assert "dspy" in answer_lower, "Answer should mention DSPy"
    assert "framework" in answer_lower, "Answer should mention 'framework'"
    assert "stanford" in answer_lower, "Answer should mention 'Stanford' (from corpus)"
    assert "language model" in answer_lower or "lm" in answer_lower, "Answer should mention language models"
    assert "structure" in answer_lower or "optimization" in answer_lower, "Answer should mention 'structure' or 'optimization' (from corpus)"
    print(f"Agent Answer (RAG): {result.answer}") # Print answer for inspection

# Test running a query that should primarily use the Wikipedia tool
@pytest.mark.integration
def test_agent_wikipedia_search(simba_agent_instance):
    print("\n--- Running test_agent_wikipedia_search ---")
    test_agent = simba_agent_instance 
    question = "Who was the first US President?"
    print(f"Querying agent: '{question}'")
    result = test_agent(question=question)

    assert result is not None, "Agent should return a result"
    assert hasattr(result, 'answer'), "Result should have an 'answer' attribute"
    assert isinstance(result.answer, str), "Answer should be a string"
    # Check if the expected answer is present (case-insensitive)
    answer_lower = result.answer.lower()
    assert "george washington" in answer_lower, "Answer should contain 'George Washington'"
    print(f"Agent Answer: {result.answer}") # Print answer for inspection
