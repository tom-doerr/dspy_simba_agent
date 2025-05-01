import logging
import typing
import dspy
import functools

# Local imports
from .model import SimpleCoder
from .evaluation import human_eval_metric
from .utils import print_code_block


def run_pre_optimization(coder: SimpleCoder, example: dspy.Example, timeout: int) -> dspy.Prediction:
    """Runs the coder on an example before optimization and logs the score."""
    prompt = example.problem["prompt"]
    task_id = example.problem.get("task_id", "UnknownTaskID")
    logging.info(f"Pre-optimization run for task {task_id}")
    result = coder(prompt=prompt)
    print_code_block(prompt + result.completion)
    score = human_eval_metric(example, result, timeout=timeout)
    logging.info(f"Score before optimization for task {task_id}: {score}")
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
    from dspy.teleprompt import SIMBA

    logging.info("Configuring SIMBA optimizer.")
    # Create a partial function for the metric with the timeout bound,
    # conforming to the (gold, pred, trace) signature expected by SIMBA.
    metric_with_timeout = functools.partial(eval_metric, timeout=timeout)

    optimizer = SIMBA(metric=metric_with_timeout, max_steps=max_steps, max_demos=max_demos)
    logging.info(f"Starting SIMBA optimization with {len(trainset)} examples...")
    optimized_coder = optimizer.compile(coder, trainset=trainset, seed=seed)
    logging.info("SIMBA optimization finished.")
    return optimized_coder


def run_post_optimization(optimized_coder: SimpleCoder, example: dspy.Example, timeout: int) -> None:
    """Runs the optimized coder on an example and logs the score."""
    prompt = example.problem["prompt"]
    task_id = example.problem.get("task_id", "UnknownTaskID")
    logging.info(f"Post-optimization run for task {task_id}")
    result = optimized_coder(prompt=prompt)
    print_code_block(prompt + result.completion)
    score = human_eval_metric(example, result, timeout=timeout)
    logging.info(f"Score after optimization for task {task_id}: {score}")
