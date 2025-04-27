"""DSPy Agent using ReAct, RAG, and SIMBA optimization."""

import argparse
import logging
import sys
import wikipedia
import dspy
from dspy.predict import ReAct
from dspy.retrievers import Embeddings
import litellm  # Import litellm
import numpy as np  # Import numpy
from typing import List, ClassVar  # Add ClassVar here

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Define Wikipedia Search Function ---
def wikipedia_search(query: str) -> str:
    """Searches Wikipedia for a given query and returns summarized results."""
    logging.info(f"--- Calling Wikipedia Function with query: '{query}' ---")
    try:
        # Use auto_suggest=False to avoid ambiguity errors on broad queries
        # Use sentences=3 for a concise summary
        results = wikipedia.summary(query, sentences=3, auto_suggest=False)
        logging.info(f"Wikipedia Result (first ~100 chars): {results[:100]}...")
        return results
    except wikipedia.exceptions.PageError:
        logging.info(f"Wikipedia page not found for query: '{query}'")
        return f"No Wikipedia page found for '{query}'."
    except wikipedia.exceptions.DisambiguationError as e:
        logging.info(
            f"Wikipedia disambiguation error for query: '{query}'. Options: {e.options[:5]}..."
        )
        return f"Query '{query}' is ambiguous on Wikipedia. Try a more specific query."
    except Exception as e:  # Catch other potential Wikipedia API errors
        logging.error(f"Error during Wikipedia search for '{query}': {e}")
        return f"An error occurred while searching Wikipedia for '{query}'."


# --- Agent Signature --- #
class AgentSignature(dspy.Signature):
    """Asks a question and provides an answer, potentially using tools or context."""

    instructions: ClassVar[str] = (
        "Asks a question and provides an answer. "
        "First, check the provided `context` to see if it contains the answer. "
        "If the `context` is sufficient, provide the `answer` based on it. "
        "If the `context` is insufficient or does not contain the answer, "
        "you may use the available tools to find the necessary information before providing the final `answer`."
    )
    context = dspy.InputField(
        desc="May contain relevant context for the question.", prefix="Context:\n"
    )
    question = dspy.InputField(desc="The question to answer.", prefix="Question: ")
    answer = dspy.OutputField(
        desc="The final answer to the question.", prefix="Answer:"
    )


# Define a callable wrapper for litellm embeddings
class LiteLLMEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts using litellm.embedding and returns a numpy array."""
        logging.debug(
            f"Embedding {len(texts)} texts with model {self.model_name} via litellm..."
        )
        response = litellm.embedding(model=self.model_name, input=texts)
        # Assuming response.data contains a list of embedding objects, each with an 'embedding' attribute
        embeddings = [item["embedding"] for item in response.data]
        logging.debug(f"Received {len(embeddings)} embeddings.")
        return np.array(embeddings)


def setup_rag(
    corpus: list[str], embedder_name: str = "openai/text-embedding-3-small", k: int = 3
):
    """Sets up the RAG retriever using FAISS and specified embeddings.

    Args:
        corpus (list[str]): The list of documents for the corpus.
        embedder_name (str): The name of the embedding model to use (e.g., from OpenAI).
        k (int): The number of top passages to retrieve.
    Returns:
        The configured retriever model, or None if setup fails.
    """
    logging.info("Setting up FAISS index for RAG...")
    try:
        # Instantiate the custom LiteLLM embedder wrapper
        logging.info(f"Creating LiteLLMEmbedder for model: {embedder_name}")
        embedding_callable = LiteLLMEmbedder(model_name=embedder_name)

        # Use dspy.retrievers.Embeddings which handles the FAISS backend
        # Pass the callable embedder instance
        retriever_model = Embeddings(corpus=corpus, k=k, embedder=embedding_callable)
        logging.info("FAISS index setup complete with k=%d.", k)
    except ImportError as exc:  # W0707: Add 'from exc'
        logging.error("FAISS library not found. Install via pip.")
        logging.error("Error: %s", exc)
        return None
    except Exception as e:
        logging.error("Error setting up FAISS: %s", e, exc_info=True)
        return None

    return retriever_model


# --- Agent Definition ---
class SimbaAgent(dspy.Module):
    """A simple ReAct agent using DSPy, optionally with RAG."""

    # Note: ReAct expects LLM and RM to be configured via dspy.settings
    def __init__(self, llm, retriever_model=None, use_tool=True):
        super().__init__()
        self.llm = llm
        self.retriever_model = retriever_model
        self.use_tool = use_tool

        # Initialize the tool only if use_tool is True
        if self.use_tool:
            # Wrap the wikipedia_search function with dspy.Tool
            self.tool = dspy.Tool(
                func=wikipedia_search,
                name="wikipedia_search",
                desc="Searches Wikipedia for a given query and returns summarized results.",
                # Let dspy.Tool infer args from the function signature
            )
        else:
            self.tool = None

    def forward(self, question):
        """Executes the agent's logic: retrieve -> generate answer.
        Uses ReAct module to decide between retrieval and tool use.
        """
        context = ""
        if self.retriever_model:
            try:
                # Retrieve relevant passages using the stored retriever model
                retrieved_docs = self.retriever_model(question).passages
                context = "\n".join(retrieved_docs)
                logging.info("Retrieved context for question: %s...", context[:100])
            except Exception as e:
                logging.error("Error during retrieval: %s", e, exc_info=True)
                context = ""  # Proceed without context if retrieval fails
        else:
            logging.info("No retriever model configured, proceeding without retrieval.")

        # Call the ReAct module with context (potentially empty)
        # ReAct will use the context and decide whether to use tools.
        tools_list = [self.tool] if self.tool else []
        result = ReAct(AgentSignature, tools=tools_list, max_iters=5)(
            question=question, context=context
        )

        # Ensure the final output is consistently a Prediction object
        if isinstance(result, dspy.Prediction):
            logging.info("ReAct module returned a Prediction.")
            return result
        # If ReAct returns a string or other type, wrap it
        logging.warning("ReAct module returned non-Prediction type: %s", type(result))
        return dspy.Prediction(answer=str(result))  # Attempt conversion


# --- Evaluation Metric ---
def metric(example, pred, trace=None):  # C0116: Added basic docstring
    """Basic evaluation metric (placeholder)."""
    # W0613: Unused 'example', 'trace' - common in dspy metrics
    # pylint: disable=unused-argument
    # Placeholder: Check if prediction is not empty
    return len(pred.answer) > 0


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Setup LLM and DSPy settings --- #
    # Configure the language model globally (Important: Do this early!)
    dspy.settings.configure(lm=dspy.LM("openrouter/google/gemini-2.0-flash-001"))

    # --- Configuration --- #
    parser = argparse.ArgumentParser(description="Run DSPy Simba Agent.")
    parser.add_argument(
        "--llm",
        type=str,
        default="openrouter/google/gemini-2.0-flash-001",
        help="LLM model name (via LiteLLM, e.g., 'openai/gpt-3.5-turbo' or 'deepseek/deepseek-chat')",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="openai/text-embedding-3-small",
        help="Embedding model name (e.g., 'openai/text-embedding-3-small')",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to the corpus file for RAG setup.",
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of passages to retrieve for RAG."
    )
    parser.add_argument(
        "--question", type=str, default=None, help="The question to ask the agent."
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run the agent in interactive mode."
    )

    args = parser.parse_args()

    # --- Setup LLM ---
    logging.info("Configuring LLM: %s", args.llm)
    llm = dspy.LM(args.llm)
    try:
        # Try a simple generation to check connectivity (optional)
        # llm("Test prompt")
        pass  # A try block cannot be empty
    except Exception as e:
        logging.error("Error configuring LLM: %s", e, exc_info=True)
        sys.exit(1)

    # --- Setup RAG (only if corpus provided) ---
    retriever_model_main = None
    if args.corpus:
        logging.info("Loading corpus from: %s", args.corpus)
        corpus_texts = []
        try:
            with open(args.corpus, "r", encoding="utf-8") as f:
                corpus_texts = [line.strip() for line in f if line.strip()]
            logging.info("Loaded %d documents from corpus.", len(corpus_texts))
            if len(corpus_texts) > 0:
                logging.info("Setting up RAG retriever...")
                try:
                    retriever_model_main = setup_rag(
                        corpus_texts, embedder_name=args.embedder, k=args.k
                    )
                    logging.info("RAG setup complete.")
                except Exception as e:
                    logging.error(
                        "Error setting up RAG retriever: %s", e, exc_info=True
                    )
                    logging.warning("Continuing without RAG due to setup error.")
            else:
                logging.warning("Corpus file loaded but was empty. No RAG setup.")
        except FileNotFoundError:
            logging.error(
                "Corpus file not found at %s. Skipping RAG setup.", args.corpus
            )
        except Exception as e:
            logging.error("Error loading corpus: %s", e, exc_info=True)

    # --- Setup Agent ---
    logging.info("Initializing Simba Agent...")
    agent = SimbaAgent(llm=llm, retriever_model=retriever_model_main)

    # --- Run Agent --- #
    if args.question:
        logging.info("Running agent with question: '%s'", args.question)
        prediction = agent(question=args.question)
        logging.info("Agent Final Answer:\n%s", prediction.answer)

        # --- Optional: Display Trace --- #
        # E1111 Fix: inspect_history likely prints, doesn't return
        llm.inspect_history(n=1)
        print("\n\n--- Agent Trace ---")
        # Trace is printed by inspect_history itself

    elif args.interactive:
        print("Entering interactive mode. Type 'quit' or 'exit' to end.")
        while True:
            try:
                user_question = input("\nEnter your question: ")
                if user_question.lower() in ["quit", "exit"]:
                    break
                if not user_question:
                    continue
                prediction = agent(question=user_question)
                print("\nAgent Answer:")
                print(prediction.answer)

                # Optionally show trace in interactive mode too
                # E1111 Fix: inspect_history likely prints, doesn't return
                llm.inspect_history(n=1)
                print("\n--- Trace ---")
                # Trace is printed by inspect_history itself

            except EOFError:
                break  # Exit on Ctrl+D
            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break  # Ensure this is indented correctly
            except Exception as e:  # W0718: Be more specific if possible
                print(f"\nAn error occurred: {e}")
                logging.error("Error in interactive loop: %s", e, exc_info=True)

    logging.info("Agent finished.")
