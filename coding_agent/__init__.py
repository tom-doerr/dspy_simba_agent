try:
    # For Python ≥3.8
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Fallback for older Python versions
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("dspy-simba-agent")
except PackageNotFoundError:
    __version__ = "0.1.0"

import dspy
import typing
import logging
import tempfile
import os
import datasets
import functools
from human_eval.data import write_jsonl, HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness
from .utils import print_code_block
from .config import configure_logging, configure_lm
from .data import load_human_eval_dataset, prepare_dspy_dataset, get_devset
from .model import SimpleCoder
from .evaluation import human_eval_metric
from .optimization import run_pre_optimization, optimize_agent, run_post_optimization

class CodingSignature(dspy.Signature):
    prompt: str = dspy.InputField(
        desc="The prompt from the HumanEval dataset (signature+docstring)."
    )
    completion: str = dspy.OutputField(
        desc="The generated Python code completion (function body)."
    )

def main(
    lm_name: str = "deepseek/deepseek-chat",
    optimization_subset_size: int = 32,
    timeout: int = 10
) -> None:
    configure_logging()
    try:
        configure_lm(lm_name)
        problems = load_human_eval_dataset()
    except Exception:
        logging.exception("Failed during setup (logging, LM config, or dataset load).")
        return

    dataset = prepare_dspy_dataset(problems)
    devset = get_devset(dataset, optimization_subset_size)
    coder = SimpleCoder()

    if devset:
        ex = devset[0]
        run_pre_optimization(coder, ex, timeout=timeout)
        try:
            opt = optimize_agent(coder, devset, human_eval_metric, max_steps=5, max_demos=1, seed=42, timeout=timeout)
            run_post_optimization(opt, ex, timeout=timeout)
        except Exception as e:
            logging.exception(f"Error during optimization or post-run: {e}")
    else:
        logging.warning("Devset empty, skipping pre/post optimization runs and optimization.")
