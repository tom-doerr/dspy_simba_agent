import pytest
from unittest.mock import patch, MagicMock
import dspy

# Adjust import based on new structure
from coding_agent.config import configure_lm

@patch('dspy.LM')
@patch('dspy.configure') 
def test_configure_lm(mock_dspy_configure, mock_dspy_lm): 
    """Test successful LM configuration."""
    lm_name = "test-model"
    # Use the actual dspy.LM class for spec, accessed via the original import
    # OR, more simply, unittest.mock can often infer from the mock object itself
    # Let's ensure the mock instance behaves like dspy.LM.
    # mock_dspy_lm is the mock class provided by the patch.
    mock_lm_instance = mock_dspy_lm.return_value # The mock instance
    # No need to create a separate MagicMock with spec here, just use the one from the patch

    # Act
    returned_lm = configure_lm(lm_name)

    # Assert
    mock_dspy_lm.assert_called_once_with(lm_name)
    # Assert configure was called with the instance returned by the mock LM class
    mock_dspy_configure.assert_called_once_with(lm=mock_lm_instance)
    # Assert the function returned the instance created by the mock LM class
    assert returned_lm == mock_lm_instance

@patch('dspy.LM')
@patch('dspy.configure')
def test_configure_lm_failure(mock_dspy_configure, mock_dspy_lm):
    """Test LM configuration failure ensures configure is not called."""
    lm_name = "invalid-model"
    mock_dspy_lm.side_effect = ValueError("Invalid model specified")
    with pytest.raises(ValueError, match="Invalid model specified"):
        configure_lm(lm_name)
    mock_dspy_configure.assert_not_called()
