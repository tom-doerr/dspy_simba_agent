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
    print("\n(Test Fixture) Setting up RAG retriever backend...")
    # setup_rag now returns the backend (e.g., Embeddings), not the client (dspy.Retrieve)
    backend_retriever = setup_rag(corpus_data)

    if backend_retriever:
        print("(Test Fixture) Configuring dspy.settings.rm globally with backend...")
        # Set the global RM to the backend instance
        dspy.settings.configure(rm=backend_retriever)
    else:
        print("(Test Fixture) RAG setup failed, dspy.settings.rm not configured.")
        # Ensure RM is explicitly unset if setup failed
        if hasattr(dspy.settings, 'rm'):
             dspy.settings.configure(rm=None)

    assert backend_retriever is not None, "RAG retriever backend setup failed in fixture"
    return backend_retriever # Return the backend instance

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
    print("\n(Test Fixture) Instantiating SimbaAgent...")
    # The agent now uses the globally configured RM, so we don't pass it here.
    # It needs the configured LLM (from setup_dspy) and the tool.
    agent = SimbaAgent(llm=dspy.settings.lm, tool=wikipedia_tool_instance)
    assert agent is not None, "SimbaAgent instantiation failed"
    return agent

# Basic test to check initialization of RAG and Agent
@pytest.mark.integration
def test_agent_initialization(rag_retriever, simba_agent_instance):
    print("\n--- Running test_agent_initialization ---")
    assert rag_retriever is not None, "RAG retriever fixture failed (backend should be created)"
    assert simba_agent_instance is not None, "SimbaAgent fixture failed"
    # Instead, check if the global RM was configured by the rag_retriever fixture
    assert dspy.settings.rm is not None, "dspy.settings.rm was not configured by fixtures"
    assert dspy.settings.rm == rag_retriever, "dspy.settings.rm is not the backend from the fixture"

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
    print(f"Agent answer: {result.answer}")

# Test running a query that should primarily use RAG
@pytest.mark.integration
def test_agent_with_rag(simba_agent_instance):
    print("\n--- Running test_agent_with_rag ---")
    test_agent = simba_agent_instance 
    question = "Explain DSPy framework based on context."
    print(f"Querying agent: '{question}'")
    # The call itself will use the globally configured RAG via dspy.Retrieve
    result = test_agent(question=question)
    assert result is not None, "Agent should return a result"
    assert hasattr(result, 'answer'), "Result should have an 'answer' attribute"
    print(f"Agent answer: {result.answer}")

# Test running a query that should primarily use the Wikipedia tool
@pytest.mark.integration
def test_agent_wikipedia_search(simba_agent_instance):
    print("\n--- Running test_agent_wikipedia_search ---")
    test_agent = simba_agent_instance 
    question = "Who was the first president of the United States?"
    print(f"Querying agent: '{question}'")
    result = test_agent(question=question)
    assert result is not None, "Agent should return a result"
    assert hasattr(result, 'answer'), "Result should have an 'answer' attribute"
    print(f"Agent answer: {result.answer}")
