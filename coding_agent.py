import dspy
import typing
import logging
import subprocess
import tempfile
import os
import datasets # Import Hugging Face datasets
from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL # Import HUMAN_EVAL constant
from human_eval.evaluation import evaluate_functional_correctness 

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # Set to DEBUG

# --- Updated Signature ---
class CodingSignature(dspy.Signature):
    """Generates Python code completion for a given HumanEval prompt."""
    # HumanEval provides a 'prompt' field which includes the signature and docstring
    prompt: str = dspy.InputField(desc="The prompt from the HumanEval dataset, including function signature and docstring.")
    # The output field should contain *only* the function body and any necessary indentation.
    # Let's guide the LM to provide just the completion.
    completion: str = dspy.OutputField(desc="The generated Python code completion (function body). Provide only the code to complete the function, starting from the first indented line.")

# --- Updated Agent ---
class SimpleCoder(dspy.Module):
    """A simple DSPy module that generates code based on a HumanEval prompt."""
    def __init__(self):
        super().__init__()
        # Using dspy.Predict with the updated signature
        self.generate_code = dspy.Predict(CodingSignature)

    def forward(self, prompt: str) -> dspy.Prediction:
        """Executes the code generation prediction."""
        prediction = self.generate_code(prompt=prompt)
        # We might need to post-process prediction.completion if the LM includes the prompt
        # Let's assume for now it generates just the completion as requested in the signature desc.
        return prediction

# --- Dataset Definition (Removed old trainset) --- 
# Dataset will be loaded in main()

# --- Metric Definition (using human_eval) ---

def human_eval_metric(gold, pred, trace=None):
    """DSPy metric for HumanEval using the human-eval library."""
    problem = gold.problem # The original HumanEval problem dict
    generated_completion = pred.completion

    with tempfile.TemporaryDirectory() as tmpdir:
        # No longer need internal read_problems() as we pass the path explicitly

        # Prepare the samples in the format the evaluator expects
        task_id = problem.get("task_id", "UnknownTaskID") # Use .get for safety
        samples = [
            dict(task_id=task_id, completion=generated_completion)
        ]
        samples_path = os.path.join(tmpdir, "samples.jsonl")

        # --- Write and Verify Samples File ---
        try:
            write_jsonl(samples_path, samples)
            logging.debug(f"human_eval_metric: Wrote samples to {samples_path}")
            if not os.path.exists(samples_path):
                 logging.error(f"human_eval_metric: FAILED TO CONFIRM EXISTENCE of {samples_path}")
                 return 0.0
            # Read back and log content for verification
            with open(samples_path, 'r') as f:
                content_read = f.read()
                logging.debug(f"human_eval_metric: Content read back from {samples_path}: {content_read.strip()}")
            if not content_read.strip():
                logging.error(f"human_eval_metric: Samples file {samples_path} is EMPTY after writing!")
                return 0.0

        except Exception as e:
            logging.error(f"human_eval_metric: Error writing or verifying samples file {samples_path}: {e}")
            return 0.0
        # --- End Verification ---

        # --- Evaluate ---
        try:
            # Evaluate functional correctness using the sample file
            results = evaluate_functional_correctness(
                sample_file=samples_path,
                problem_file=HUMAN_EVAL, # Explicitly provide the problem file path
                k=[1],
                timeout=10  # seconds timeout
            )

            pass_at_1 = results.get("pass@1", 0.0) # Use .get for safety
            if pass_at_1 > 0:
                logging.debug(f"Metric: Tests passed for task_id: {task_id}")
                return 1.0
            else:
                # Use debug level for failures as they are expected during optimization
                logging.debug(f"Metric: Tests failed for task_id: {task_id}")
                return 0.0

        except Exception as e:
            # Log the full error and the code that caused it
            logging.error(f"Metric: Error during file-based human_eval execution for task_id: {task_id}. Error: {e}")
            # Also log the sample file content again in case of error
            try:
                with open(samples_path, 'r') as f_err:
                    content_err = f_err.read()
                    logging.error(f"Content of {samples_path} during error: {content_err.strip()}")
            except Exception as read_err:
                logging.error(f"Could not read samples file {samples_path} during error handling: {read_err}")
            logging.error(f"Problem Dict: {problem}")
            logging.error(f"Attempted Code:\n---\n{generated_completion}\n---")
            return 0.0 # Return 0.0 on error


def main():
    # --- Configuration --- 
    # Switch to deepseek/deepseek-chat
    lm_name = "deepseek/deepseek-chat" 
    lm = dspy.LM(lm_name)
    dspy.configure(lm=lm)
    logging.info(f"Configuring LM: {lm_name}")

    # --- Load HumanEval Data --- 
    logging.info("Loading HumanEval dataset via Hugging Face datasets...")
    # This loads the full dataset. 
    try:
        # Use the correct dataset identifier from Hugging Face Hub
        human_eval_ds = datasets.load_dataset("openai_humaneval") 
        # The dataset likely has a 'test' split containing the problems
        if 'test' not in human_eval_ds:
            raise ValueError("Expected 'test' split not found in openai_humaneval dataset.")
        human_eval_problems = list(human_eval_ds['test'])
        logging.info(f"Loaded {len(human_eval_problems)} HumanEval problems.")
    except Exception as e:
        logging.error(f"Failed to load HumanEval dataset: {e}")
        logging.error("Please ensure 'datasets' and 'human-eval' are installed.")
        return
 
     # Convert to dspy.Example format, storing the original problem dict
     # We only need the 'prompt' as input for the agent.
    dspy_dataset = [
        dspy.Example(prompt=item['prompt'], problem=item).with_inputs('prompt')
        for item in human_eval_problems
    ]
    
    # For demonstration, let's use a smaller subset for optimization
    # You might use human_eval_dataset.train and human_eval_dataset.test later if available
    # Or split manually: train_size = int(0.8 * len(dspy_dataset))
    optimization_subset_size = 32 # Use 32 examples for optimization demo (SIMBA requirement)
    if len(dspy_dataset) < optimization_subset_size:
        logging.warning(f"Dataset size ({len(dspy_dataset)}) is smaller than requested optimization subset size ({optimization_subset_size}). Using full dataset.")
        devset = dspy_dataset
    else:
        devset = dspy_dataset[:optimization_subset_size]
        # testset = dspy_dataset[optimization_subset_size:] # Optional: for final evaluation
    logging.info(f"Using {len(devset)} examples for optimization/dev.")

    # --- Initialize Agent --- 
    coder = SimpleCoder()
    logging.info("Initialized SimpleCoder Agent.")

    # --- Example Usage (without optimization yet) --- 
    if devset:
        example_problem_for_run = devset[0].problem # Get the corresponding problem dict
        example_prompt = example_problem_for_run['prompt']
        
        logging.info(f"--- Running agent BEFORE optimization with task_id: {example_problem_for_run['task_id']} ---")
        result_before = coder(prompt=example_prompt)
        logging.info("Agent Generated Completion (Before Optimization):")
        print("```python")
        # Print the full function for context
        print(example_prompt + result_before.completion)
        print("```")
        # Evaluate the unoptimized code
        score_before = human_eval_metric(devset[0], result_before)
        logging.info(f"Score before optimization (pass@1): {score_before}")
    else:
        logging.warning("Development set is empty, skipping pre-optimization run.")


    # --- Optimization Setup --- 
    from dspy.teleprompt import SIMBA
    
    # Configure the optimizer using the new metric
    # max_demos=1 might be suitable for code generation where examples are long
    simba_optimizer = SIMBA(metric=human_eval_metric, max_steps=5, max_demos=1) 
    logging.info("Configured SIMBA optimizer.")

    # --- Run Optimization --- 
    if devset:
        logging.info(f"Starting SIMBA optimization with {len(devset)} examples...")
        try:
            optimized_coder = simba_optimizer.compile(coder, trainset=devset, seed=42)
            logging.info("SIMBA optimization finished.")

            # --- Example Usage (AFTER optimization) --- 
            logging.info(f"--- Running agent AFTER optimization with task_id: {example_problem_for_run['task_id']} ---")
            result_after = optimized_coder(prompt=example_prompt)
            logging.info("Agent Generated Completion (After Optimization):")
            print("```python")
            # Print the full function for context
            print(example_prompt + result_after.completion)
            print("```")
            # Evaluate the optimized code
            score_after = human_eval_metric(devset[0], result_after)
            logging.info(f"Score after optimization (pass@1): {score_after}")
            
            # Save the optimized program 
            # optimized_coder.save("optimized_coder_humaneval.json")
            # logging.info("Saved optimized coder model.")
        except Exception as e:
            logging.error(f"Error during SIMBA optimization: {e}")
            # If optimization fails, optimized_coder might not be defined.
            # Consider adding fallback or stopping execution.
    else:
        logging.warning("Development set is empty, skipping optimization and post-optimization run.")


if __name__ == "__main__":
    main()
