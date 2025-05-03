#!/usr/bin/env python
# tests/test_action_executor.py
import xml.etree.ElementTree as ET
from unittest.mock import Mock

import pytest

# Assume simpledspy_agent is importable (adjust path if needed, e.g., via sys.path or project structure)
from simpledspy_agent import ActionExecutor, AgentIO

# --- Fixtures ---


@pytest.fixture
def mock_io_handler() -> Mock:
    """Provides a mocked AgentIO handler."""
    return Mock(spec=AgentIO)


@pytest.fixture
def action_executor(mock_io_handler: Mock) -> ActionExecutor:
    """Provides an ActionExecutor instance with a mocked IO handler."""
    return ActionExecutor(io_handler=mock_io_handler)


# --- Test Cases ---


def test_handle_think(action_executor: ActionExecutor, mock_io_handler: Mock):
    """Test that _handle_think calls io_handler.display_thought."""
    xml_string = (
        "<action type='think'><content> Pondering the meaning... </content></action>"
    )
    action_element = ET.fromstring(xml_string)

    result = action_executor._handle_think(action_element)

    assert result == (False, None)  # Think action should return False
    mock_io_handler.display_thought.assert_called_once_with("Pondering the meaning...")


def test_handle_message(action_executor: ActionExecutor, mock_io_handler: Mock):
    """Test that _handle_message calls io_handler.display_message."""
    xml_string = "<action type='message'><content> Hello User! </content></action>"
    action_element = ET.fromstring(xml_string)

    result = action_executor._handle_message(action_element)

    assert result == (True, None)  # Message action should return True
    mock_io_handler.display_message.assert_called_once_with("Agent: Hello User!")


def test_handle_run_command_with_cwd(
    action_executor: ActionExecutor, mock_io_handler: Mock
):
    """Test _handle_run_command with command and CWD."""
    xml_string = """
    <action type='run_command'>
        <command_line>ls -l</command_line>
        <cwd>/tmp</cwd>
    </action>
    """
    action_element = ET.fromstring(xml_string)

    result = action_executor._handle_run_command(action_element)

    assert result  # Run command action should return True
    mock_io_handler.display_command_request.assert_called_once_with("ls -l", "/tmp")


def test_handle_run_command_no_cwd(
    action_executor: ActionExecutor, mock_io_handler: Mock
):
    """Test _handle_run_command with command but no CWD."""
    xml_string = "<action type='run_command'><command_line>pwd</command_line></action>"
    action_element = ET.fromstring(xml_string)

    result = action_executor._handle_run_command(action_element)

    assert result
    mock_io_handler.display_command_request.assert_called_once_with("pwd", None)


def test_handle_run_command_no_command(
    action_executor: ActionExecutor, mock_io_handler: Mock
):
    """Test _handle_run_command with no command_line element."""
    xml_string = "<action type='run_command'><cwd>/tmp</cwd></action>"
    action_element = ET.fromstring(xml_string)

    result = action_executor._handle_run_command(action_element)

    assert result == (False, None)  # Should return False if no command
    mock_io_handler.display_command_request.assert_not_called()
    mock_io_handler.display_message.assert_called_once_with(
        "Agent: (run_command action requested but no command_line found)"
    )


def test_execute_action_valid(action_executor: ActionExecutor, mock_io_handler: Mock):
    """Test execute_action dispatching to a valid handler (message)."""
    xml_string = "<action type='message'><content>Dispatch test</content></action>"
    action_element = ET.fromstring(xml_string)

    result = action_executor.execute_action("message", action_element)

    assert result == (True, None)
    mock_io_handler.display_message.assert_called_once_with("Agent: Dispatch test")


def test_execute_action_invalid_type(
    action_executor: ActionExecutor, mock_io_handler: Mock
):
    """Test execute_action with an unknown action type."""
    xml_string = "<action type='fly_to_moon'><content>Engage!</content></action>"
    action_element = ET.fromstring(xml_string)

    result = action_executor.execute_action("fly_to_moon", action_element)

    assert result == (False, None)
    mock_io_handler.display_message.assert_called_once_with(
        "Agent: (Unknown action type: fly_to_moon)"
    )


def test_execute_action_handler_error(
    action_executor: ActionExecutor, mock_io_handler: Mock
):
    """Test execute_action when the handler itself raises an exception."""
    # Make the mock display_message raise an error when called
    mock_io_handler.display_message.side_effect = ValueError("Mock display failed")

    xml_string = "<action type='message'><content>This will fail</content></action>"
    action_element = ET.fromstring(xml_string)

    result = action_executor.execute_action("message", action_element)

    assert result == (False, None)
    mock_io_handler.display_message.assert_called_once_with(
        "Agent: This will fail"
    )  # It was called
    mock_io_handler.display_error.assert_called_once_with(
        "Executing action 'message': Mock display failed"
    )
