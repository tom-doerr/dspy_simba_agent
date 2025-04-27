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
# 1. Define a dummy corpus (replace with actual documents later)
dummy_corpus = [
    "The first DSPy agent used a ReAct module for tool use.",
    "Simba is an optimizer in DSPy for improving prompts and instructions.",
    "RAG combines retrieval with generation to answer questions using external knowledge.",
    "Faiss is a library for efficient similarity search, often used for vector stores.",
    "The 2022 Nobel Peace Prize went to Ales Bialiatski, Memorial, and Center for Civil Liberties."
]

# 2. Set up the Retriever Model (using in-memory EmbedRetriever)
# Ensure OPENAI_API_KEY is set in the environment for embeddings
try:
    # Using OpenAI embeddings as suggested in the plan
    embedder = Embedder("openai/text-embedding-3-small")
    retriever_model = Embeddings(embedder=embedder, corpus=dummy_corpus, k=3)
    dspy.settings.configure(rm=retriever_model, lm=llm) # Configure retriever globally along with LM
    print("RAG Retriever configured successfully using OpenAI embeddings.")
except Exception as e:
    print(f"Failed to configure OpenAI Embedder/Retriever: {e}")
    print("RAG will not be available.")
    # Optionally configure a fallback or exit
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
    ("What does RAG stand for?", "RAG combines retrieval with generation to answer questions using external knowledge."),
    ("What is Faiss?", "Faiss is a library for efficient similarity search, often used for vector stores."),
    ("What is the role of the ReAct module in the initial agent?", "The first DSPy agent used a ReAct module for tool use."),
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

    if retriever_model is None:
        print("Skipping agent test as retriever is not configured.")
    else:
        result = agent(question=question)
        print("\nAgent Result:")
        print(f"Answer: {result.answer}")

        # Print the full history for debugging/inspection
        print("\nAgent Trace:")
        # Inspect history of the underlying LLM
        # Need to access the LM used by the react_module, assuming global settings
        # or potentially access via agent.react_module.signature.lm if configured per module
        dspy.settings.lm.inspect_history(n=1) # Inspect the last interaction

    # Run Simba Optimization
    print("\n--- Starting Simba Optimization ---")
    # Check if retriever is available, as RAG context might be important for optimization
    if retriever_model:
        # Instantiate the optimizer
        # Note: max_steps and max_demos are small for quick testing
        simba_optimizer = dspy.SIMBA(metric=validate_answer, max_steps=3, max_demos=4, bsize=4) # Set bsize

        # Compile the agent
        # student=agent copies the agent structure
        optimized_agent = simba_optimizer.compile(student=agent, trainset=trainset, seed=123)
        
        print("\n--- Optimization Complete ---")

        # Test the optimized agent
        print(f"\nTesting OPTIMIZED agent with question: '{question}'")
        opt_result = optimized_agent(question=question)
        print("\nOptimized Agent Result:")
        print(f"Answer: {opt_result.answer}")
        print("\nOptimized Agent Trace:")
        dspy.settings.lm.inspect_history(n=1)
    else:
        print("Skipping optimization as retriever is not configured.")
