import dspy
import wikipedia
import os

# RAG Imports
from dspy.retrieve import Retrieve
from dspy.retrievers import Embeddings
from dspy import Embedder
import numpy as np # May still be needed indirectly

# Configure DSPy LM - using a model specified in user rules
# llm = dspy.OpenAI(model='gpt-3.5-turbo') # Example from plan
# Using user-specified model preference
llm = dspy.LM('openrouter/google/gemini-2.0-flash-001')
dspy.settings.configure(lm=llm)

def search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia for a query and return a list of relevant page titles."""
    try:
        results = wikipedia.search(query, results=5)  # Get top 5 titles
        return results
    except wikipedia.exceptions.WikipediaException as e:
        print(f"Wikipedia search error: {e}")
        return []

def lookup_page(title: str) -> str:
    """Look up a specific Wikipedia page by title and return its summary."""
    try:
        # Fetch the page summary, limit to a few sentences for brevity
        page = wikipedia.page(title, auto_suggest=False) # Disable auto_suggest for exact match
        summary = wikipedia.summary(title, sentences=3) # Get first 3 sentences
        # Truncate summary to avoid overloading context (e.g., 500 chars)
        max_chars = 500
        return summary[:max_chars] + ('...' if len(summary) > max_chars else '')
    except wikipedia.exceptions.PageError:
        return f"Could not find a Wikipedia page titled '{title}'."
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation by returning options or a message
        # For simplicity, just return a message indicating ambiguity
        return f"'{title}' is ambiguous. Options include: {e.options[:3]}..."
    except wikipedia.exceptions.WikipediaException as e:
        print(f"Wikipedia lookup error: {e}")
        return "Error retrieving page content."

# --- RAG Setup ---
# 1. Define the Corpus
corpus_path = "corpus.txt"
corpus = []
if os.path.exists(corpus_path):
    with open(corpus_path, 'r') as f:
        corpus = [line.strip() for line in f if line.strip()] # Read non-empty lines
    print(f"Loaded corpus from {corpus_path} with {len(corpus)} documents.")
else:
    print(f"Warning: Corpus file '{corpus_path}' not found. RAG will have no knowledge base.")

# 2. Set up the Retriever Model (using in-memory EmbedRetriever)
# Ensure OPENAI_API_KEY is set in the environment for embeddings
try:
    # Using OpenAI embeddings as suggested in the plan
    embedder = Embedder("openai/text-embedding-3-small")
    # Use the loaded corpus
    if corpus: # Only create retriever if corpus is not empty
        retriever_model = Embeddings(embedder=embedder, corpus=corpus, k=3)
        dspy.settings.configure(rm=retriever_model, lm=llm) # Configure retriever globally along with LM
        print("RAG Retriever configured successfully using OpenAI embeddings and corpus.txt.")
    else:
        print("Skipping retriever configuration as corpus is empty or file not found.")
except Exception as e:
    print(f"Error setting up RAG Retriever: {e}")
    print("RAG functionality will be disabled.")
    retriever_model = None # Indicate retriever is unavailable
    dspy.settings.configure(lm=llm) # Configure just the LM

# --- Agent Definition ---
# 1. Define the ReAct agent's I/O signature, now including context
class RAGToolUseSignature(dspy.Signature):
    """Answer questions using the retrieved context and the tools [search_wikipedia, lookup_page]. Search iteratively if needed and provide a concise answer based on context and tool results."""
    context: list[str] = dspy.InputField(desc="Facts retrieved from the knowledge base.")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# 2. Instantiate the ReAct module (renamed for clarity)
react_module = dspy.ReAct(
    RAGToolUseSignature,
    tools=[search_wikipedia, lookup_page],
    max_iters=5 # Limit the number of tool calls/steps
)

# 3. Define the main SimbaAgent module combining RAG and ReAct
class SimbaAgent(dspy.Module):
    def __init__(self, retriever_model, react_module):
        super().__init__()
        # Store the retriever model directly
        self.retriever_model = retriever_model
        self.react_module = react_module

    def forward(self, question):
        # Call the retriever model directly
        if self.retriever_model:
            # Assuming the retriever model instance itself is callable and returns passages
            # Check dspy.retrievers.Embeddings documentation if needed
            # The retriever model should use the 'k' it was initialized with
            retrieved_data = self.retriever_model(question)
            context = retrieved_data.passages if hasattr(retrieved_data, 'passages') else retrieved_data
            # Filter out potential None values
            context = [c for c in context if c is not None] 
        else:
            context = [] # No context if retriever is disabled
        
        # Pass context and question to the ReAct module
        result = self.react_module(context=context, question=question)
        return result

# Instantiate the main agent
agent = SimbaAgent(retriever_model=retriever_model, react_module=react_module)

# --- Simba Optimization Setup ---

# 1. Create a simple trainset (replace with a real dataset later)
train_data = [
    ("What framework is mentioned for programming LMs?", "DSPy is a framework from Stanford for programming language models."),
    ("What is Faiss?", "Faiss is a library for efficient similarity search, often used for vector stores."),
    ("What does the ReAct paradigm combine?", "ReAct is an agent paradigm that combines reasoning and acting."),
    ("Who received the 2022 Nobel Peace Prize?", "The 2022 Nobel Peace Prize went to Ales Bialiatski, Memorial, and Center for Civil Liberties.")
]

# Convert to dspy.Example objects
# Assuming RAG will provide context, we focus on question -> answer mapping for simplicity here
# For more complex optimization, the trainset examples could include expected context/tool usage
trainset = [dspy.Example(question=q, answer=a).with_inputs('question') for q, a in train_data]

# 2. Define an evaluation metric
def validate_answer(example, pred, trace=None):
    """Basic metric: checks if the predicted answer contains the gold answer substring."""
    # Simple substring check - can be improved (e.g., LLM-based check, F1 score)
    predicted_answer = pred.answer.lower()
    gold_answer = example.answer.lower()
    return gold_answer in predicted_answer

# --- Test / Optimization Execution ---

# Test the agent (pre-optimization)
if __name__ == "__main__":
    # Test with a question relevant to the dummy corpus
    # question = "Who won the Nobel Peace Prize in 2022?" # Previous test
    question = "What is Simba in DSPy?"
    print(f"Testing agent with question: '{question}'")

    # Run the agent without optimization
    result = agent(question=question)
    print("\nAgent Result:")
    print(f"Answer: {result.answer}")
    print("\nAgent Trace:")
    if hasattr(dspy.settings.lm, 'inspect_history'):
        dspy.settings.lm.inspect_history(n=1) # Inspect the last interaction

    # --- Simba Optimization --- 
    optimized_agent_path = "simba_agent_optimized.json"
    optimized_agent = None

    try:
        # Try to load the optimized agent
        agent_to_load = SimbaAgent(retriever_model=None, react_module=None) # Need a base structure to load into
        agent_to_load.load(optimized_agent_path)
        # Re-assign the actual submodules after loading the state
        agent_to_load.retriever_model = retriever_model # Use the model created earlier
        agent_to_load.react_module = react_module     # Use the module created earlier
        optimized_agent = agent_to_load
        print(f"\nLoaded optimized agent from {optimized_agent_path}")
    except FileNotFoundError:
        print(f"\nOptimized agent file '{optimized_agent_path}' not found. Running optimization...")
        
        print("\n--- Starting Simba Optimization ---")
        # Check if retriever is available, as RAG context might be important for optimization
        # Also check if corpus was loaded successfully
        if retriever_model and corpus:
            # Instantiate the optimizer
            # Note: max_steps and max_demos are small for quick testing
            simba_optimizer = dspy.SIMBA(metric=validate_answer, max_steps=3, max_demos=4, bsize=4) # Set bsize

            # Compile the agent
            # student=agent copies the agent structure
            compiled_agent = simba_optimizer.compile(student=agent, trainset=trainset, seed=123)
            
            print("\n--- Optimization Complete ---")

            # Save the optimized agent
            compiled_agent.save(optimized_agent_path)
            print(f"Saved optimized agent to {optimized_agent_path}")
            optimized_agent = compiled_agent # Use the newly compiled agent
        else:
            print("Skipping optimization as retriever is not configured or corpus is empty.")
            # Fall back to the unoptimized agent if optimization skipped
            optimized_agent = agent 

    # Test the final agent (either loaded or newly optimized)
    if optimized_agent:
        print(f"\nTesting FINAL agent with question: '{question}'")
        opt_result = optimized_agent(question=question)
        print("\nFinal Agent Result:")
        print(f"Answer: {opt_result.answer}")
        print("\nFinal Agent Trace:")
        if hasattr(dspy.settings.lm, 'inspect_history'):
            dspy.settings.lm.inspect_history(n=1)
