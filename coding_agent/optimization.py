import logging
import typing
import dspy
from dspy.teleprompt import SIMBA
import functools

# Local imports
from .model import SimpleCoder
from .utils import print_code_block
from human_eval.execution import check_correctness


def run_pre_optimization(coder: SimpleCoder, example: dspy.Example, timeout: int) -> dspy.Prediction:
    """Runs the coder on an example before optimization and evaluates it."""
    logging.info(f"Pre-optimization run for task {example.problem.get('task_id', 'N/A')}")
    result = coder(prompt=example.problem['prompt'])
    print_code_block(result.completion)
    # Evaluate using check_correctness for single problem
    eval_result = check_correctness(problem=example.problem, completion=result.completion, timeout=timeout)
    pass_status = "PASSED" if eval_result['passed'] else "FAILED"
    logging.info(f"Pre-optimization evaluation result: {pass_status}")
    return result


def optimize_agent(
    coder: SimpleCoder,
    trainset: typing.List[dspy.Example],
    eval_metric, # The actual evaluation function (e.g., human_eval_metric)
    max_steps: int,
    max_demos: int,
    seed: int,
    timeout: int # Timeout for the evaluation metric during optimization
) -> dspy.Module:
    """Optimizes the coder using dspy.teleprompt.SIMBA."""
    logging.info("Configuring SIMBA optimizer.")
    # Create a partial function for the metric with the timeout bound,
    # conforming to the (gold, pred, trace) signature expected by SIMBA.
    metric_with_timeout = functools.partial(eval_metric, timeout=timeout)

    # Configure the SIMBA optimizer
    optimizer = SIMBA(
        metric=metric_with_timeout,
        max_steps=max_steps,
        max_demos=max_demos,
        # bsize=len(trainset) # Explicitly set batch size - causes high memory usage
        bsize=16 # Use a smaller fixed batch size
    )

    logging.info(f"Starting SIMBA optimization with {len(trainset)} examples...")
    optimized_coder = optimizer.compile(coder, trainset=trainset, seed=seed)
    logging.info("SIMBA optimization finished.")
    return optimized_coder


def run_post_optimization(optimized_coder: SimpleCoder, example: dspy.Example, timeout: int) -> None:
    """Runs the optimized coder on an example and evaluates it."""
    logging.info(f"Post-optimization run for task {example.problem.get('task_id', 'N/A')}")
    result = optimized_coder(prompt=example.problem['prompt'])
    print_code_block(result.completion)
    # Evaluate using check_correctness for single problem
    eval_result = check_correctness(problem=example.problem, completion=result.completion, timeout=timeout)
    pass_status = "PASSED" if eval_result['passed'] else "FAILED"
    logging.info(f"Post-optimization evaluation result: {pass_status}")
