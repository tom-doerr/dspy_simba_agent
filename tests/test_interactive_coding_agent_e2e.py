# tests/test_simpledspy_agent_e2e.py
import pytest
from unittest.mock import MagicMock, call
import xml.etree.ElementTree as ET

# Assuming the agent and components are importable
# Adjust the import path if your project structure is different
from simpledspy_agent import SimpleAgent
from agent_components import AgentIO, ActionExecutor, ConsoleIO

# --- Mock Components ---

class MockAgentIO(AgentIO):
    def __init__(self, inputs=None):
        self.inputs = inputs if inputs else []
        self.outputs = []
        self.input_index = 0

    def get_input(self, prompt: str) -> str:
        if self.input_index < len(self.inputs):
            user_input = self.inputs[self.input_index]
            self.outputs.append(f"InputPrompt: {prompt}")
            self.outputs.append(f"UserInput: {user_input}")
            self.input_index += 1
            return user_input
        raise EOFError("No more mock input")

    def display_message(self, message: str):
        self.outputs.append(f"Message: {message}")

    def display_thought(self, thought: str):
        self.outputs.append(f"Thought: {thought}")

    def display_command_request(self, command: str, cwd: str | None):
        cwd_str = f" in {cwd}" if cwd else ""
        self.outputs.append(f"CommandRequest: {command}{cwd_str}")

    def display_error(self, error_message: str):
        self.outputs.append(f"Error: {error_message}")

    def display_warning(self, warning_message: str):
        self.outputs.append(f"Warning: {warning_message}")

    def display_raw_response(self, raw_response: str):
        # We might not need to track this explicitly in outputs for e2e
        pass # self.outputs.append(f"RawResponse: {raw_response[:50]}...")

class MockActionExecutor(ActionExecutor):
    def __init__(self, io_handler: AgentIO):
        super().__init__(io_handler)
        # Store calls for assertion
        self.executed_actions = []
        # Predefine command outputs
        self.command_outputs = {}

    def set_command_output(self, command_line: str, output: str):
        self.command_outputs[command_line] = output

    def execute_action(self, action_type: str, action_element: ET.Element) -> tuple[bool, str | None]:
        action_xml_str = ET.tostring(action_element, encoding='unicode').strip() # Strip whitespace
        self.executed_actions.append((action_type, action_xml_str))

        if action_type == "run_command":
            command_element = action_element.find("command_line")
            cwd_element = action_element.find("cwd") # Also get cwd for display
            command_line = command_element.text.strip() if command_element is not None and command_element.text else None
            cwd = cwd_element.text.strip() if cwd_element is not None and cwd_element.text else None # Extract cwd
            if command_line and command_line in self.command_outputs:
                output = self.command_outputs[command_line]
                # Simulate the IO display that the real executor does
                self.io.display_command_request(command_line, cwd) # <--- Add this call
                self.io.display_message("--- Command Output ---")
                self.io.display_message(output) # Simplified display for mock
                self.io.display_message("--- End Command Output ---")
                return True, output # Return success and the predefined output
            else:
                self.io.display_error(f"MockExecutor: No predefined output for {command_line}")
                return False, None
        elif action_type == "message":
            content_element = action_element.find("content")
            content = content_element.text.strip() if content_element is not None and content_element.text else ""
            self.io.display_message(f"Agent: {content}") # Simulate message display
            return True, None
        elif action_type == "think":
            content_element = action_element.find("content")
            content = content_element.text.strip() if content_element is not None and content_element.text else ""
            self.io.display_thought(content) # Simulate thought display
            return False, None
        else:
            self.io.display_message(f"Agent: (Unknown action type: {action_type}) - Mocking as no-op")
            return False, None # Not a recognized visible action, no command output

# --- Test Fixture --- 

@pytest.fixture
def mock_agent(mocker):
    """Provides a SimpleAgent instance with mocked IO, Executor, and LLM calls."""
    mock_io = MockAgentIO()
    mock_executor = MockActionExecutor(mock_io)

    # Create agent instance with mocks
    agent = SimpleAgent(io_handler=mock_io, action_executor=mock_executor, model_name="mock_model")

    # Mock the _get_llm_response method directly on the instance
    mock_llm_response = mocker.patch.object(agent, '_get_llm_response', autospec=True)
    agent.mock_llm_response = mock_llm_response # Attach mock to agent for easy access in tests

    # Mock _configure_lm to always succeed
    mocker.patch.object(agent, '_configure_lm', return_value=True)

    return agent, mock_io, mock_executor


# --- Test Cases ---

def test_simple_greeting(mock_agent, mocker):
    """Test a simple interaction without commands."""
    agent, mock_io, mock_executor = mock_agent

    # Configure mock inputs and LLM responses
    mock_io.inputs = ["hello", "quit"]
    agent.mock_llm_response.side_effect = [
        # Response to 'hello'
        "<response><actions><action type='think'><content>User said hello.</content></action><action type='message'><content>Hi there!</content></action></actions></response>",
        # Response to 'quit' (though loop should exit before LLM call)
    ]

    # Run the agent
    agent.run()

    # Assertions
    initial_welcome = f"Message: \nSimple DSPy Agent ({agent.model_name}) - Type 'quit' to exit."
    assert mock_io.outputs == [
        initial_welcome,
        'InputPrompt: User: ', 
        'UserInput: hello',
        'Thought: User said hello.',
        'Message: Agent: Hi there!',
        'InputPrompt: User: ', 
        'UserInput: quit',
        'Message: \nAgent session finished.'
    ]
    assert agent.mock_llm_response.call_count == 1
    agent.mock_llm_response.assert_called_once_with("hello")
    assert mock_executor.executed_actions == [
        ('think', '<action type="think"><content>User said hello.</content></action>'), 
        ('message', '<action type="message"><content>Hi there!</content></action>')
    ]

def test_multi_step_command_execution(mock_agent, mocker):
    """Test the agent running a command and using its output for a second LLM call."""
    agent, mock_io, mock_executor = mock_agent

    # --- Test Setup ---
    user_request = "how much disk space?"
    command_to_run = "df -h /"
    simulated_command_output = "Filesystem Size Used Avail Use% Mounted on\n/dev/sda1 100G 60G 40G 60% /"
    final_answer = "Your root filesystem (/) has 40G available (60% used)."

    # 1. Mock User Input
    mock_io.inputs = [user_request, "quit"]

    # 2. Mock Executor Command Output
    mock_executor.set_command_output(command_to_run, simulated_command_output)

    # 3. Mock LLM Responses (Two calls expected)
    first_llm_response = f"""
<response>
  <actions>
    <action type='think'><content>User wants disk space. Run df -h.</content></action>
    <action type='run_command'><command_line>{command_to_run}</command_line></action>
  </actions>
</response>
"""
    second_llm_response = f"""
<response>
  <actions>
    <action type='think'><content>Command output received. Extract relevant info and reply.</content></action>
    <action type='message'><content>{final_answer}</content></action>
  </actions>
</response>
"""
    agent.mock_llm_response.side_effect = [first_llm_response, second_llm_response]

    # --- Run Agent ---
    agent.run()

    # --- Assertions ---

    # 1. Check IO interactions sequence
    expected_io_outputs = [
        f"Message: \nSimple DSPy Agent ({agent.model_name}) - Type 'quit' to exit.", # Add initial welcome
        f'InputPrompt: User: ', # Initial prompt
        f'UserInput: {user_request}',
        'Thought: User wants disk space. Run df -h.', # First thought
        f'CommandRequest: {command_to_run}', # Command request
        'Message: --- Command Output ---',      # Command output display
        f'Message: {simulated_command_output}',
        'Message: --- End Command Output ---',
        'Thought: Command executed and returned output. Asking LLM for final response based on output...', # Second thought (internal)
        'Thought: Command output received. Extract relevant info and reply.', # Second thought (from LLM)
        f'Message: Agent: {final_answer}', # Final message
        'InputPrompt: User: ', # Prompt for next input
        'UserInput: quit',
        'Message: \nAgent session finished.'
    ]
    assert mock_io.outputs == expected_io_outputs

    # 2. Check LLM calls
    assert agent.mock_llm_response.call_count == 2
    expected_follow_up_prompt = (
        f"Original user request: {user_request}\n\n"
        f"I previously ran a command and got the following output:\n"
        f"{simulated_command_output}\n\n"
        f"Based on the original request and this output, what is the final response or next action?"
    )
    agent.mock_llm_response.assert_has_calls([
        call(user_request),
        call(expected_follow_up_prompt)
    ])

    # 3. Check executed actions
    expected_actions = [
        # First response actions
        ('think', '<action type="think"><content>User wants disk space. Run df -h.</content></action>'),
        ('run_command', f'<action type="run_command"><command_line>{command_to_run}</command_line></action>'),
        # Second response actions
        ('think', '<action type="think"><content>Command output received. Extract relevant info and reply.</content></action>'),
        ('message', f'<action type="message"><content>{final_answer}</content></action>')
    ]
    assert mock_executor.executed_actions == expected_actions
