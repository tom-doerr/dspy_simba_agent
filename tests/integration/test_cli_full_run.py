import sys
from coding_agent.__main__ import main_cli
# Import modules containing functions to be mocked
import coding_agent.config
import coding_agent.data
import coding_agent.evaluation

def test_cli_full_run(monkeypatch):
    # Target functions in their new modules
    monkeypatch.setattr(coding_agent.config, "configure_lm", lambda lm_name: None)
    # Provide dummy data to trigger optimization/evaluation steps
    monkeypatch.setattr(coding_agent.data, "load_human_eval_dataset", lambda: [{'prompt': 'P', 'task_id': 'TID'}])
    # Mock the problematic evaluation function to prevent the assertion error
    # It needs to accept the args: sample_file, problem_file, k, timeout
    monkeypatch.setattr(coding_agent.evaluation, "evaluate_functional_correctness", 
                        lambda sample_file, problem_file, k, timeout: {"pass@1": 0.5})

    # Simulate invocation with no flags (uses defaults)
    monkeypatch.setattr(sys, "argv", ["coding-agent"])
    # Should execute the main logic without raising the assertion error
    main_cli()
    # Add an assertion here if needed, e.g., check logs or outputs if possible
    # For now, the test passes if main_cli() runs without error.
