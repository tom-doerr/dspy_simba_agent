import pytest
import sys
import logging
from unittest.mock import patch, MagicMock
import dspy

# Adjust imports for submodules
import coding_agent.config
import coding_agent.data
import coding_agent.evaluation
import coding_agent.model
import coding_agent.optimization
from coding_agent.main_logic import main

def test_cli_full_run(monkeypatch, caplog):
    # Mock key components to avoid external calls and complex setup

    # Monkeypatch configure_logging first to avoid it interfering with caplog
    monkeypatch.setattr(coding_agent.main_logic, "configure_logging", lambda: None)

    # --- Refined Mocking Setup ---
    # 1. Create a mock for the LM instance
    mock_lm_instance = MagicMock(spec=dspy.LM)

    # 2. Create a mock for the response object
    mock_response = MagicMock() # Don't need spec=dspy.Prediction if we set attrs
    mock_response.completion = "mocked completion string" # Ensure this is a string

    # 3. Configure the mock LM instance to return the mock response when called
    mock_lm_instance.return_value = mock_response # Set return_value on the instance itself

    # Patch dspy.configure and dspy.LM (using the pre-configured instance)
    with patch('dspy.configure') as mock_dspy_configure, \
         patch('dspy.LM', return_value=mock_lm_instance) as mock_dspy_lm_class, \
         patch('coding_agent.main_logic.load_human_eval_dataset') as mock_load_dataset, \
         patch('coding_agent.main_logic.prepare_dspy_dataset') as mock_prepare_dataset, \
         patch('coding_agent.main_logic.optimize_agent') as mock_optimize_agent, \
         patch('coding_agent.evaluation.human_eval_metric') as mock_eval_metric, \
         patch('coding_agent.main_logic.run_pre_optimization') as mock_run_pre, \
         patch('coding_agent.main_logic.run_post_optimization') as mock_run_post, \
         patch('coding_agent.optimization.check_correctness') as mock_check_correctness: # check_correctness is likely called *within* optimization/evaluation funcs, so patch original

        # Mock check_correctness return value if needed by pre/post runs
        mock_check_correctness.return_value = {'passed': True}

        # 1. Config - No need to mock configure_lm directly anymore
        # Mock logging setup if needed, or let caplog handle capture
        # monkeypatch.setattr(coding_agent.config, "configure_logging", lambda: None)

        # 2. Data Loading (Keep existing mocks)
        mock_dataset = [
            dspy.Example(problem={'task_id': 't1', 'prompt': 'p1', 'entry_point': 'e1', 'canonical_solution': 'cs1', 'test': 'tst1'}, solution='s1').with_inputs('prompt'),
            dspy.Example(problem={'task_id': 't2', 'prompt': 'p2', 'entry_point': 'e2', 'canonical_solution': 'cs2', 'test': 'tst2'}, solution='s2').with_inputs('prompt'),
            dspy.Example(problem={'task_id': 't3', 'prompt': 'p3', 'entry_point': 'e3', 'canonical_solution': 'cs3', 'test': 'tst3'}, solution='s3').with_inputs('prompt'),
            dspy.Example(problem={'task_id': 't4', 'prompt': 'p4', 'entry_point': 'e4', 'canonical_solution': 'cs4', 'test': 'tst4'}, solution='s4').with_inputs('prompt')
        ]
        mock_load_dataset.return_value = [{'task_id': 'HumanEval/0', 'prompt': 'p1', 'entry_point': 'e1', 'canonical_solution': 'cs1', 'test': 'tst1'}]
        mock_prepare_dataset.return_value = mock_dataset
        # Mock get_devset using setattr and a mock with side_effect
        mock_get_devset = MagicMock(side_effect=lambda dataset, size: dataset[:size])
        monkeypatch.setattr(coding_agent.main_logic, "get_devset", mock_get_devset) # Patch in main_logic

        # 3. Model
        # Mock the SimpleCoder class's __init__ or forward if needed, but SIMBA itself is mocked.
        # SimpleCoder instance will be created inside main_logic using the mocked dspy.LM.

        # 4. Evaluation Metric (Keep existing mock)
        mock_eval_metric.return_value = 0.5

        # 5. Optimization (Keep existing mocks)
        # Mock the result of the SIMBA compile call
        mock_coder_instance_after_compile = MagicMock(spec=coding_agent.model.SimpleCoder)
        mock_optimizer = MagicMock()
        mock_optimizer.compile.return_value = mock_coder_instance_after_compile # Optimizer returns a coder instance
        # Patch the SIMBA class to return our mock optimizer instance when called
        monkeypatch.setattr(coding_agent.optimization, "SIMBA", lambda metric, max_steps, max_demos, bsize: mock_optimizer)
        # Mock the pre/post optimization functions directly

        # Mock data loading
        mock_prepare_dataset.return_value = mock_dataset

        # Mock Optimization result
        mock_optimized_coder = MagicMock(spec=dspy.Module) # Mock the coder itself
        # Create the mock result that the coder should return when called
        mock_coder_result = MagicMock()
        mock_coder_result.completion = "mocked optimized completion string"
        # Configure the mock coder to return this result object when called
        mock_optimized_coder.return_value = mock_coder_result
        # Configure the mock optimize_agent function to return the mock coder
        mock_optimize_agent.return_value = mock_optimized_coder

        # 6. Execute the main function directly
        caplog.set_level(logging.INFO) # Capture INFO level logs
        main(
            lm_name="mock_lm", # This name will be passed to the mocked dspy.LM
            optimization_subset_size=2, # Corresponds to --subset-size=2
            timeout=5 # Corresponds to --timeout=5 (example)
        )

        # Assertions using captured logs
        log_text = caplog.text
        # print("CAPTURED LOGS:\n", log_text) # For debugging

        # Assert logs (as before)
        assert "Configuring LM" in log_text
        # assert "Loading HumanEval dataset" in log_text # Removed as function is fully mocked now
        # Removed: assert "Preparing DSPy dataset" in log_text
        assert "Using example task t1 for pre/post runs." in log_text # Reverted based on mock_dataset

        # Check if key mocked functions were called
        mock_dspy_configure.assert_called_once() # Check dspy.configure was called by configure_lm
        mock_dspy_lm_class.assert_called_once_with("mock_lm") # Check dspy.LM was called by configure_lm
        # Verify the lm instance passed to configure matches the one returned by LM
        assert mock_dspy_configure.call_args[1]['lm'] == mock_lm_instance

        mock_load_dataset.assert_called_once()
        mock_prepare_dataset.assert_called_once() # Assert that the prepare_dspy_dataset mock was called
        mock_get_devset.assert_called_once() # Assert the mock get_devset was called
        mock_run_pre.assert_called_once()
        # Check the example passed to mock_run_pre
        example_arg_pre = mock_run_pre.call_args[0][1]
        assert example_arg_pre.problem['task_id'] == 't1' # Reverted based on mock_dataset

        mock_optimize_agent.assert_called_once()
        mock_run_post.assert_called_once()
        # Check the example passed to mock_run_post
        example_arg_post = mock_run_post.call_args[0][1]
        assert example_arg_post.problem['task_id'] == 't1' # Reverted based on mock_dataset
