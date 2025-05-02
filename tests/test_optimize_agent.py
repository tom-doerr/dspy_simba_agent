import pytest
import dspy
from unittest.mock import Mock, patch, MagicMock
import typing

# Adjust import based on new structure
from coding_agent.optimization import optimize_agent
from coding_agent.model import SimpleCoder

@pytest.fixture
def mock_coder() -> Mock:
    return MagicMock(spec=SimpleCoder)

@pytest.fixture
def mock_metric() -> Mock:
    # A simple metric that returns 1.0
    return MagicMock(return_value=1.0)

@pytest.fixture
def mock_trainset() -> typing.List[dspy.Example]:
    """Provides a mock training set (list of mock dspy.Examples)."""
    examples = []
    # Create enough examples, SIMBA uses bsize=16 internally
    for i in range(16):
        # Use MagicMock for Example objects to handle potential attribute access
        example_mock = MagicMock(spec=dspy.Example)
        # Mock attributes commonly accessed by optimizers/metrics
        # Provide basic structure, adapt if specific attributes are needed
        example_mock.problem = {'task_id': f'task_{i}', 'prompt': f'prompt_{i}'} # Mock problem dict
        example_mock.solution = f'solution_{i}' # Mock solution string
        # Mock the .with_inputs behavior if necessary (often used)
        example_mock.with_inputs.return_value = example_mock

        # If specific input/output fields are accessed directly:
        # example_mock.prompt = f'prompt_{i}'
        # example_mock.completion = f'completion_{i}' # Or whatever output field is expected

        examples.append(example_mock)
    return examples

@patch('coding_agent.optimization.SIMBA') # Try patching where it's used
def test_optimize_agent(mock_simba_class, mock_coder, mock_trainset, mock_metric):
    """Test the main optimization logic."""
    max_steps = 5
    max_demos = 3
    seed = 123
    timeout = 60

    # Configure the mock SIMBA instance
    mock_optimizer_instance = MagicMock() # Use MagicMock for the instance too
    mock_optimizer_instance.compile.return_value = MagicMock(spec=SimpleCoder) # Mock the compile result
    mock_simba_class.return_value = mock_optimizer_instance

    # Call the function under test
    optimized_coder_result = optimize_agent(
        coder=mock_coder,
        trainset=mock_trainset,
        eval_metric=mock_metric,
        max_steps=max_steps,
        max_demos=max_demos,
        seed=seed,
        timeout=timeout
    )

    # Assert SIMBA was configured correctly
    mock_simba_class.assert_called_once()
    # Check metric, max_steps, max_demos - bsize is tricky due to fixed value vs len(trainset)
    # Instead of checking bsize=len(mock_trainset), we check the fixed value
    args, kwargs = mock_simba_class.call_args
    assert kwargs['metric'] is not None # Check the partial function was passed
    assert kwargs['max_steps'] == max_steps
    assert kwargs['max_demos'] == max_demos
    assert kwargs['bsize'] == 16 # Assert the fixed batch size

    # Assert compile was called correctly
    mock_optimizer_instance.compile.assert_called_once_with(
        mock_coder, trainset=mock_trainset, seed=seed
    )

    # Assert the result is the compiled coder
    assert optimized_coder_result == mock_optimizer_instance.compile.return_value
