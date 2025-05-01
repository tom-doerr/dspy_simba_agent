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
from human_eval.data import write_jsonl, HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

# Print a Python code block
def print_code_block(code: str) -> None:
    print("```python")
    print(code)
    print("```")

def configure_logging(level: int = logging.DEBUG) -> None:
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

def configure_lm(lm_name: str) -> dspy.LM:
    logging.info(f"Configuring LM: {lm_name}")
    lm = dspy.LM(lm_name)
    dspy.configure(lm=lm)
    return lm

def load_human_eval_dataset() -> typing.List[dict]:
    logging.info("Loading HumanEval dataset via Hugging Face datasets...")
    try:
        human_eval_ds = datasets.load_dataset("openai_humaneval")
        if "test" not in human_eval_ds:
            raise ValueError("Expected 'test' split not found in openai_humaneval dataset.")
        problems = list(human_eval_ds["test"])
        logging.info(f"Loaded {len(problems)} HumanEval problems.")
        return problems
    except Exception as e:
        logging.error(f"Failed to load HumanEval dataset: {e}")
        logging.error("Please ensure 'datasets' and 'human-eval' are installed.")
        raise

def prepare_dspy_dataset(problems: typing.List[dict]) -> typing.List[dspy.Example]:
    return [
        dspy.Example(prompt=item["prompt"], problem=item).with_inputs("prompt")
        for item in problems
    ]

def get_devset(dataset: typing.List[dspy.Example], optimization_subset_size: int) -> typing.List[dspy.Example]:
    if len(dataset) < optimization_subset_size:
        logging.warning(
            f"Dataset size ({len(dataset)}) < subset size ({optimization_subset_size}), using full dataset."
        )
        return dataset
    devset = dataset[:optimization_subset_size]
    logging.info(f"Using {len(devset)} examples for optimization/dev.")
    return devset

class CodingSignature(dspy.Signature):
    prompt: str = dspy.InputField(
        desc="The prompt from the HumanEval dataset (signature+docstring)."
    )
    completion: str = dspy.OutputField(
        desc="The generated Python code completion (function body)."
    )

class SimpleCoder(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generate_code = dspy.Predict(CodingSignature)

    def forward(self, prompt: str) -> dspy.Prediction:
        return self.generate_code(prompt=prompt)

def human_eval_metric(gold, pred, trace=None) -> float:
    problem = gold.problem
    gen = pred.completion
    task_id = problem.get("task_id", "UnknownTaskID")

    with tempfile.TemporaryDirectory() as tmpdir:
        samples = [{"task_id": task_id, "completion": gen}]
        samples_path = os.path.join(tmpdir, "samples.jsonl")

        try:
            write_jsonl(samples_path, samples)
            logging.debug(f"Wrote samples to {samples_path}")
            if not os.path.exists(samples_path):
                logging.error(f"{samples_path} missing after write.")
                return 0.0
            if not open(samples_path).read().strip():
                logging.error(f"{samples_path} is empty.")
                return 0.0
        except Exception as e:
            logging.error(f"Error writing samples file: {e}")
            return 0.0

        try:
            results = evaluate_functional_correctness(
                sample_file=samples_path, problem_file=HUMAN_EVAL, k=[1], timeout=10
            )
            return 1.0 if results.get("pass@1", 0) > 0 else 0.0
        except Exception as e:
            logging.error(f"Evaluation error for task {task_id}: {e}")
            try:
                logging.error("Samples content:\n" + open(samples_path).read())
            except Exception:
                pass
            logging.error(f"Problem: {problem}")
            logging.error(f"Attempted code:\n{gen}")
            return 0.0

def run_pre_optimization(coder: SimpleCoder, example: dspy.Example) -> dspy.Prediction:
    prompt = example.problem["prompt"]
    logging.info(f"Pre-optimization run for task {example.problem.get('task_id')}")
    result = coder(prompt=prompt)
    print_code_block(prompt + result.completion)
    score = human_eval_metric(example, result)
    logging.info(f"Score before optimization: {score}")
    return result

def optimize_agent(
    coder: SimpleCoder,
    trainset: typing.List[dspy.Example],
    metric,
    max_steps: int,
    max_demos: int,
    seed: int,
) -> dspy.Module:
    from dspy.teleprompt import SIMBA

    logging.info("Configuring SIMBA optimizer.")
    optimizer = SIMBA(metric=metric, max_steps=max_steps, max_demos=max_demos)
    logging.info(f"Starting optimization with {len(trainset)} examples.")
    optimized = optimizer.compile(coder, trainset=trainset, seed=seed)
    logging.info("Optimization finished.")
    return optimized

def run_post_optimization(optimized: SimpleCoder, example: dspy.Example) -> None:
    prompt = example.problem["prompt"]
    logging.info(f"Post-optimization run for task {example.problem.get('task_id')}")
    result = optimized(prompt=prompt)
    print_code_block(prompt + result.completion)
    score = human_eval_metric(example, result)
    logging.info(f"Score after optimization: {score}")

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
        return

    dataset = prepare_dspy_dataset(problems)
    devset = get_devset(dataset, optimization_subset_size)
    coder = SimpleCoder()

    if devset:
        ex = devset[0]
        run_pre_optimization(coder, ex)
        try:
            opt = optimize_agent(coder, devset, human_eval_metric, max_steps=5, max_demos=1, seed=42)
            run_post_optimization(opt, ex)
        except Exception as e:
            logging.error(f"Error during optimization: {e}")
    else:
        logging.warning("Devset empty, skipping runs.")
