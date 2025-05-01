import logging
import tempfile
import os
import dspy
from human_eval.data import write_jsonl, HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

def human_eval_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, timeout: int = 10) -> float:
    """Evaluates the functional correctness of a code generation prediction
    using the HumanEval benchmark setup.

    Args:
        gold: The ground truth dspy.Example, containing the problem dict.
        pred: The prediction dspy.Prediction, containing the generated completion.
        trace: Optional trace information (unused in this implementation).
        timeout: The timeout in seconds for functional evaluation.

    Returns:
        1.0 if the prediction passes the evaluation, 0.0 otherwise.

    Raises:
        IOError: If there are issues creating or writing the temporary samples file.
        Exception: Re-raises exceptions from evaluate_functional_correctness.
    """
    if not hasattr(gold, 'problem') or not isinstance(gold.problem, dict):
        logging.error("Golden example missing 'problem' attribute or it's not a dict.")
        raise ValueError("Golden example must have a 'problem' dict attribute.")

    problem = gold.problem
    gen = pred.completion
    task_id = problem.get("task_id", "UnknownTaskID")

    with tempfile.TemporaryDirectory() as tmpdir:
        samples = []
        if gen: # Only create samples if code was generated
             samples = [{"task_id": task_id, "completion": gen}]

        samples_path = os.path.join(tmpdir, "samples.jsonl")

        try:
            if samples: # Only write if there are samples
                write_jsonl(samples_path, samples)
                logging.debug(f"Wrote samples to {samples_path} for task {task_id}")
                if not os.path.exists(samples_path):
                    logging.error(f"{samples_path} missing after write for task {task_id}.")
                    raise IOError(f"{samples_path} missing after write for task {task_id}.")
                if not open(samples_path).read().strip():
                    logging.error(f"{samples_path} is empty after write for task {task_id}.")
                    raise IOError(f"{samples_path} is empty after write for task {task_id}.")
            else:
                logging.debug(f"No completion generated for task {task_id}. No samples file created.")

        except Exception as e:
            logging.error(f"Error writing samples file for task {task_id}: {e}")
            raise

        # If no samples were generated, the score is 0
        if not samples:
             logging.debug(f"No samples to evaluate for task {task_id}, score is 0.")
             return 0.0

        # If samples were expected but file doesn't exist (should be caught earlier, but double-check)
        if not os.path.exists(samples_path):
             logging.error(f"Samples file {samples_path} not found for task {task_id} before evaluation.")
             raise IOError(f"Samples file {samples_path} not found for task {task_id} before evaluation.")

        try:
            results = evaluate_functional_correctness(
                sample_file=samples_path, problem_file=HUMAN_EVAL, k=[1], timeout=timeout
            )
            pass_rate = results.get("pass@1", 0)
            logging.debug(f"Evaluation result for task {task_id}: pass@1 = {pass_rate}")
            return 1.0 if pass_rate > 0 else 0.0
        except Exception as e:
            logging.error(f"Evaluation error for task {task_id}: {e}")
            try:
                # Log samples content only if file exists
                if os.path.exists(samples_path):
                    logging.error(f"Samples content for task {task_id}:\n" + open(samples_path).read())
                else:
                    logging.error(f"Samples file {samples_path} did not exist for task {task_id} upon evaluation error.")
            except Exception as read_err:
                logging.error(f"Could not read samples file {samples_path} for task {task_id} after evaluation error: {read_err}")
            logging.error(f"Problem details for task {task_id}: {problem}")
            logging.error(f"Attempted code for task {task_id}:\n{gen}")
            raise
