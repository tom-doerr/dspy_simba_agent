import logging
import os
import tempfile
import typing
import dspy

from human_eval.data import HUMAN_EVAL, write_jsonl
from human_eval.execution import check_correctness


def human_eval_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, timeout: int = 10) -> float:
    """DSPy metric compatible with HumanEval check_correctness.

    Args:
        gold (dspy.Example): The gold example containing the problem dictionary.
        pred (dspy.Prediction): The prediction containing the completion.
        trace: Optional trace object (unused).
        timeout (int): Timeout for code execution.

    Returns:
        float: 1.0 if the completion passes the check, 0.0 otherwise.
    """
    if not hasattr(gold, 'problem') or not isinstance(gold.problem, dict):
        logging.error("Gold example missing 'problem' dictionary.")
        return 0.0
    if not hasattr(pred, 'completion') or not isinstance(pred.completion, str):
        logging.error("Prediction object missing 'completion' string.")
        return 0.0

    problem = gold.problem
    completion = pred.completion
    task_id = problem.get("task_id", "unknown_task")

    try:
        # Use check_correctness for single problem evaluation
        eval_result = check_correctness(
            problem=problem,
            completion=completion,
            timeout=timeout
        )
        score = 1.0 if eval_result['passed'] else 0.0
        # Optional: Log pass/fail status
        # status = "PASSED" if score == 1.0 else "FAILED"
        # logging.debug(f"Evaluation for {task_id}: {status}")
        return score

    except Exception as e:
        logging.error(f"Evaluation error for task {task_id}: {e}")
        # Optionally log more details
        # logging.error(f"Problem details for task {task_id}: {problem}")
        # logging.error(f"Attempted code for task {task_id}:\n{completion}")
        return 0.0 # Return 0.0 on any evaluation error
