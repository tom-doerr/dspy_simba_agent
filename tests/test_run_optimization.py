import pytest
from unittest.mock import patch, MagicMock
import dspy
import coding_agent.optimization as ca_opt # Use alias for clarity
import logging # Import logging

# Dummy classes for testing
class DummyCoder:
    def __init__(self):
        self.forward_called = False
    def __call__(self, prompt):
        self.forward_called = True
        return MagicMock(completion="XYZ") # Return a mock with completion

class DummyExample:
    def __init__(self):
        self.problem = {'task_id': 'dummy_task', 'prompt': 'dummy prompt'}

# Test functions

def test_run_pre_optimization(capsys, caplog, monkeypatch): # Add caplog
    # Mock check_correctness within the optimization module namespace
    monkeypatch.setattr(ca_opt, "check_correctness", lambda problem, completion, timeout=10: {'passed': True})
    coder = DummyCoder()
    example = DummyExample()
    caplog.set_level(logging.INFO) # Set log level for capture

    # Pass timeout
    result = ca_opt.run_pre_optimization(coder, example, timeout=10)
    out = capsys.readouterr().out # Capture stdout/stderr
    log_text = caplog.text # Capture log messages

    # Check print output via capsys
    assert " 1 " in out
    assert "XYZ" in out # Check for the completion content

    # Check logging output via caplog
    assert "Pre-optimization run for task dummy_task" in log_text
    assert "Pre-optimization evaluation result: PASSED" in log_text

    # Check return value and other side effects
    assert result.completion == "XYZ" # Check the returned result
    assert coder.forward_called

def test_run_post_optimization(capsys, caplog, monkeypatch): # Add caplog
    # Mock check_correctness within the optimization module namespace
    monkeypatch.setattr(ca_opt, "check_correctness", lambda problem, completion, timeout=10: {'passed': False})
    coder = DummyCoder()
    example = DummyExample()
    caplog.set_level(logging.INFO) # Set log level for capture

    # should not return anything, only print
    # Pass timeout
    ret = ca_opt.run_post_optimization(coder, example, timeout=10)
    out = capsys.readouterr().out # Capture stdout/stderr
    log_text = caplog.text # Capture log messages

    assert ret is None
    # Check print output via capsys
    assert " 1 " in out
    assert "XYZ" in out # Check for the completion content

    # Check logging output via caplog
    assert "Post-optimization run for task dummy_task" in log_text
    assert "Post-optimization evaluation result: FAILED" in log_text
    assert coder.forward_called # Ensure coder was called
