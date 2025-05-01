import typing
import logging
import dspy
import datasets

# Note: HUMAN_EVAL is used in the evaluation metric, not directly here.

def load_human_eval_dataset() -> typing.List[dict]:
    """Loads the HumanEval dataset using the Hugging Face datasets library."""
    logging.info("Loading HumanEval dataset via Hugging Face datasets...")
    try:
        # This returns the list directly when split='test' is provided
        human_eval_ds = datasets.load_dataset("openai_humaneval", split="test")
        logging.info(f"Loaded {len(human_eval_ds)} problems from HumanEval.")
        # The check below is no longer needed as load_dataset(split='test') either
        # returns the list for the 'test' split or raises an error if the split doesn't exist.
        return list(human_eval_ds) # Ensure it's a list
    except Exception:
        logging.exception("Failed to load HumanEval dataset.")
        logging.error("Please ensure 'datasets' library is installed and you have network access.")
        raise

def prepare_dspy_dataset(problems: typing.List[dict]) -> typing.List[dspy.Example]:
    """Converts the raw HumanEval problems into dspy.Example objects."""
    return [
        dspy.Example(prompt=item["prompt"], problem=item).with_inputs("prompt")
        for item in problems
    ]

def get_devset(dataset: typing.List[dspy.Example], optimization_subset_size: int) -> typing.List[dspy.Example]:
    """Selects a subset of the dataset to use as the development set for optimization."""
    if len(dataset) < optimization_subset_size:
        logging.warning(
            f"Dataset size ({len(dataset)}) < subset size ({optimization_subset_size}), using full dataset."
        )
        return dataset
    devset = dataset[:optimization_subset_size]
    logging.info(f"Using {len(devset)} examples for optimization/dev.")
    return devset
