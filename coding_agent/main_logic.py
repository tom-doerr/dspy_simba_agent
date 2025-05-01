import logging

# Local imports from the refactored modules
from .config import configure_logging, configure_lm
from .data import load_human_eval_dataset, prepare_dspy_dataset, get_devset
from .model import SimpleCoder
from .evaluation import human_eval_metric
from .optimization import run_pre_optimization, optimize_agent, run_post_optimization

def main(
    lm_name: str = "deepseek/deepseek-chat",
    optimization_subset_size: int = 32,
    timeout: int = 10
) -> None:
    """Main orchestration logic for the DSPy SIMBA coding agent.

    Configures logging and LM, loads data, prepares dataset,
    initializes the coder, runs pre-optimization, optimization,
    and post-optimization steps.
    """
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
        # Use first example for pre/post runs for demonstration
        # A more robust approach might evaluate on the whole devset or a sample
        example_for_runs = devset[0]
        logging.info(f"Using example task {example_for_runs.problem.get('task_id', 'N/A')} for pre/post runs.")

        run_pre_optimization(coder, example_for_runs, timeout=timeout)
        try:
            # Note: Hardcoded SIMBA parameters (max_steps, max_demos, seed)
            # These could be exposed as CLI args later.
            optimized_coder = optimize_agent(
                coder=coder,
                trainset=devset,
                eval_metric=human_eval_metric,
                max_steps=5,
                max_demos=1,
                seed=42,
                timeout=timeout
            )
            run_post_optimization(optimized_coder, example_for_runs, timeout=timeout)
        except Exception as e:
            logging.exception(f"Error during optimization or post-run: {e}")
    else:
        logging.warning("Development set is empty, skipping optimization and pre/post runs.")
