import dspy
import wikipedia
import os

# RAG Imports
from dspy.evaluate.metrics import answer_exact_match # Use exact match for now

# Define necessary components (classes and functions) globally

# --- RAG Setup (Helper Functions) ---
def load_corpus(filepath: str) -> list[str]:
    """Loads the text corpus from a file, one document per line."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f if line.strip()]
        if not corpus:
            print(f"Warning: Corpus file '{filepath}' is empty or contains only whitespace.")
            return []
        print(f"Loaded corpus from {filepath} with {len(corpus)} documents.")
        return corpus
    except FileNotFoundError:
        print(f"Error: Corpus file not found at '{filepath}'.")
        return []
    except Exception as e:
        print(f"Error loading corpus from '{filepath}': {e}")
        return []

def setup_rag(corpus: list[str], embedder_name: str = "openai/text-embedding-3-small", dimensions: int = 512, k: int = 3) -> dspy.Module:
    """Sets up the RAG retriever using dspy.retrievers.Embeddings.

    Args:
        corpus: A list of document strings.
        embedder_name: Name of the embedding model to use.
        dimensions: The dimension size of the embeddings.
        k: The default number of passages to retrieve.

    Returns:
        An instance of dspy.retrievers.Embeddings (or a similar backend) or None on error.
    """
    if not corpus:
        print("Cannot setup RAG without a loaded corpus.")
        return None # Return None if corpus is empty

    try:
        print(f"Setting up Embeddings retriever with {embedder_name} ({dimensions}d) for {len(corpus)} docs...")
        # Instantiate the embedder
        embedder = dspy.Embedder(embedder_name, dimensions=dimensions)

        # Instantiate the Embeddings retriever backend
        # Note: This might require `faiss-cpu` installed if the corpus is large
        # The 'k' here is the default, but can be overridden during the call
        backend_retriever = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=k)

        print("Embeddings Retriever backend configured successfully.")
        return backend_retriever
    except ModuleNotFoundError as e:
        if 'faiss' in str(e):
            print("Error setting up RAG: FAISS library not found. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'.")
            print("Attempting to continue without FAISS (may be slow or fail for large corpora). Consider installing FAISS.")
            # Potentially try brute force if FAISS is missing, but dspy.retrievers.Embeddings might handle this
            try:
                embedder = dspy.Embedder(embedder_name, dimensions=dimensions)
                # Check if brute force is an option or if it handles it internally
                backend_retriever = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=k)
                print("Embeddings Retriever backend configured successfully (likely using brute force).")
                return backend_retriever
            except Exception as inner_e:
                print(f"Error setting up RAG even without FAISS attempt: {inner_e}")
                return None
        else:
            print(f"Error setting up RAG (ModuleNotFoundError): {e}")
            return None
    except Exception as e:
        print(f"Error setting up RAG: {e}")
        return None

# --- Tool Setup (Helper Function) ---
def search_wikipedia(query: str) -> list[str]:
    """Searches Wikipedia and returns the first paragraph of the top 3 results."""
    try:
        results = wikipedia.search(query, results=3)
        summaries = []
        for title in results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                first_paragraph = next((p for p in page.content.split('\n') if p.strip()), "")
                summaries.append(f"Title: {page.title}\nSummary: {first_paragraph}")
            except wikipedia.exceptions.PageError:
                summaries.append(f"Title: {title}\nSummary: Could not load page details.")
            except wikipedia.exceptions.DisambiguationError as e:
                summaries.append(f"Title: {title}\nSummary: Disambiguation error. Options: {e.options[:5]}")
        return summaries
    except Exception as e:
        print(f"Error during Wikipedia search: {e}")
        return [f"Error searching Wikipedia: {e}"]

# --- Agent Definition ---
class ReActSignature(dspy.Signature):
    """Define the input/output behavior of the ReAct module."""
    context = dspy.InputField(desc="May contain relevant context.")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Often a detailed response to the question.")

class ReActModule(dspy.Module):
    """The ReAct module that uses the wikipedia tool."""
    def __init__(self, tools: list):
        super().__init__()
        self.react = dspy.ReAct(ReActSignature, tools=tools)

    def forward(self, question, context=None):
        return self.react(question=question, context=context)

class SimbaAgent(dspy.Module):
    """The main agent integrating RAG and ReAct."""
    def __init__(self, llm, tool, k=3):
        super().__init__()
        self.k = k # Number of passages to retrieve
        self.react_module = ReActModule(tools=[tool]) # Instantiate ReActModule here

    def forward(self, question):
        # Use dspy.Retrieve locally, which relies on globally configured dspy.settings.rm
        retrieved_docs_str = ""
        try:
            # Create a local Retrieve client that uses the global rm
            retrieve_client = dspy.Retrieve(k=self.k)
            retrieved_docs_list = retrieve_client(question).passages
            retrieved_docs_str = "\n".join(retrieved_docs_list)
            print(f"\nRetrieved {len(retrieved_docs_list)} passages for question: '{question}'")
        except Exception as e:
            print(f"\nError during retrieval for question '{question}': {e}")
            print("Proceeding without retrieved context.")
            # Potentially check if dspy.settings.rm is configured
            if not dspy.settings.rm:
                print("Warning: dspy.settings.rm is not configured.")

        # Call the ReAct module with context (potentially empty)
        result = self.react_module(question=question, context=retrieved_docs_str)

        return dspy.Prediction(answer=result.answer)

    def save(self, path):
        print(f"Attempting to save ReAct module state to {path}...")
        # Only save the ReAct module's state, as it contains the compiled reasoning trace
        if hasattr(self, 'react_module') and self.react_module:
            self.react_module.save(path)
            print(f"Saved ReAct module state to {path}")

    def load(self, path):
        print(f"Attempting to load agent state (ReAct module) from {path}...")
        # Load the state into the existing react_module
        if hasattr(self, 'react_module') and self.react_module:
            try:
                self.react_module.load(path)
                print(f"Loaded ReAct module state from {path}")
            except Exception as e:
                print(f"Error loading ReAct module state from {path}: {e}")
        else:
            print("Error: react_module not initialized before load.")

# --- Evaluation Metric ---
# 1. Define dummy data (replace with actual data later)
dummy_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
    # Add more diverse examples relevant to your potential corpus/tasks
    {"question": "Explain DSPy Simba", "answer": "Simba is an optimizer in DSPy for few-shot learning in agents or multi-step programs."},
    {"question": "What year did the Titanic sink?", "answer": "1912"}
]
trainset = [dspy.Example(x).with_inputs('question') for x in dummy_data]

# 2. Define an evaluation metric
def validate_answer(example, pred, trace=None):
    """Validates the predicted answer against the gold answer using exact match."""
    # Reverting to exact match for simplicity after import error
    is_correct = answer_exact_match(example, pred, trace=trace)
    print(f"Gold: {example.answer} | Pred: {pred.answer} | Exact Match: {is_correct}")
    return is_correct # answer_exact_match returns True/False

# --- Test / Optimization Execution --- Only runs when script is executed directly

if __name__ == "__main__":
    import argparse # Import here as it's only needed for direct execution
    from dotenv import load_dotenv # Import here

    load_dotenv() # Load environment variables

    # --- Instantiate DSPy components inside the main block ---
    print("(Main) Instantiating and configuring DSPy components...")
    llm = dspy.LM("openrouter/google/gemini-2.0-flash-001") # User preference
    print("(Main) Configuring DSPy settings...")
    dspy.settings.configure(lm=llm)

    # Instantiate the Wikipedia tool
    wikipedia_tool = dspy.Tool(name="wikipedia_search",
                               desc="Searches Wikipedia for a given query.",
                               input_variable="query",
                               func=search_wikipedia)

    # --- Set up RAG --- 
    corpus_filepath = "corpus.txt" # Define corpus path
    corpus = load_corpus(corpus_filepath)
    retriever_model = setup_rag(corpus) if corpus else None
    if not retriever_model:
        print("Warning: RAG setup failed or corpus is empty. Agent will run without retrieval.")

    # --- Instantiate the Agent ---
    agent = SimbaAgent(llm=llm, tool=wikipedia_tool)

    # --- Argument Parsing for command-line execution ---
    parser = argparse.ArgumentParser(description="Run or optimize the DSPy Simba Agent.")
    parser.add_argument("--query", type=str, default="What is DSPy Simba?", help="The question to ask the agent.")
    parser.add_argument("--optimize", action="store_true", help="Run Simba optimization.")
    parser.add_argument("--load_opt", type=str, default="simba_agent_optimized.json", help="Path to load optimized agent state.")
    parser.add_argument("--save_opt", type=str, default="simba_agent_optimized.json", help="Path to save optimized agent state.")
    args = parser.parse_args()

    agent_to_run = agent # Default to the non-optimized agent

    # --- Load or Optimize --- 
    optimized_agent_path = args.load_opt
    if os.path.exists(optimized_agent_path) and not args.optimize:
        try:
            print(f"\nLoading optimized agent from {optimized_agent_path}...")
            # Re-instantiate the agent structure before loading
            loaded_agent = SimbaAgent(llm=llm, tool=wikipedia_tool)
            loaded_agent.load(optimized_agent_path) # Load state into the react_module
            agent_to_run = loaded_agent
            print("Successfully loaded optimized agent state.")
        except Exception as e:
            print(f"Error loading optimized agent: {e}. Using unoptimized agent.")
            agent_to_run = agent

    elif args.optimize:
        print("\n--- Starting Simba Optimization ---")
        if retriever_model and corpus and wikipedia_tool:
            simba_optimizer = dspy.SIMBA(metric=validate_answer, max_steps=3, max_demos=4, bsize=4)
            print("Starting Simba optimization with validate_answer metric...")
            try:
                # Ensure the student agent has the correct submodules before compiling
                student_agent = SimbaAgent(llm=llm, tool=wikipedia_tool)
                compiled_agent = simba_optimizer.compile(student=student_agent, trainset=trainset, seed=123)
                print("\n--- Optimization Complete ---")
                agent_to_run = compiled_agent
                save_path = args.save_opt
                agent_to_run.save(save_path)
                print(f"Saved optimized agent to {save_path}")
            except Exception as e:
                print("\n--- ERROR during Simba Optimization --- ")
                print(f"{type(e).__name__}: {e}")
                print("Optimization failed. Using unoptimized agent.")
                agent_to_run = agent # Fallback to unoptimized
        else:
            print("Skipping optimization as RAG components (retriever/corpus) or tools are not ready.")
            agent_to_run = agent # Use unoptimized if components missing

    # --- Execute Query --- 
    print(f"\n--- Running Agent ({'Loaded Optimized' if agent_to_run is not agent and not args.optimize else ('Newly Optimized' if args.optimize else 'Unoptimized')}) --- ")
    test_query = args.query
    print(f"Query: {test_query}")
    final_result = agent_to_run(question=test_query)
    print(f"\nFinal Answer: {final_result.answer}")

    # Optional: Evaluate after running
    # print("\n--- Evaluating Final Agent on Trainset --- ")
    # evaluate = Evaluate(devset=trainset, metric=validate_answer, num_threads=1, display_progress=True)
    # score = evaluate(agent_to_run)
    # print(f"\nFinal Agent Score (Exact Match on trainset): {score}")
