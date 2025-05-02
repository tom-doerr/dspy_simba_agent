# simpledspy_agent.py
from simpledspy import pipe
import dspy
import xml.etree.ElementTree as ET
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
import argparse
from rich.console import Console
from rich.prompt import Prompt

# --- IO Abstraction ---

class AgentIO(ABC):
    """Abstract base class for Agent Input/Output handling."""
    @abstractmethod
    def get_input(self, prompt: str) -> str:
        pass

    @abstractmethod
    def display_message(self, message: str, end: str = '\n'):
        pass

    @abstractmethod
    def display_thought(self, thought: str):
        pass

    @abstractmethod
    def display_command_request(self, command_line: str, cwd: str | None):
        pass

    @abstractmethod
    def display_error(self, error_message: str):
        pass

    @abstractmethod
    def display_raw_response(self, raw_response: str):
        pass

class ConsoleIO(AgentIO):
    """Concrete implementation of AgentIO using the console."""
    def __init__(self):
        self.console = Console()

    def get_input(self, prompt: str) -> str:
        return Prompt.ask(f"[bold green]{prompt}[/bold green]")

    def display_message(self, message: str, end: str = '\n'):
        self.console.print(message, end=end)

    def display_thought(self, thought: str):
        self.console.print(f"Agent (Thought): {thought}")

    def display_command_request(self, command_line: str, cwd: str | None):
        self.console.print(f"Agent wants to run command:")
        self.console.print(f"  Command: {command_line}")
        if cwd:
            self.console.print(f"  CWD: {cwd}")
        self.console.print("  (Command execution is currently disabled for safety)")

    def display_error(self, error_message: str):
        self.console.print(f"Error: {error_message}", style="red")

    def display_raw_response(self, raw_response: str):
        self.console.print("\n--- Raw XML Response ---")
        self.console.print(raw_response)
        self.console.print("------------------------\n")

# --- Action Executor Class ---

class ActionExecutor:
    """Handles the execution of specific actions requested by the agent."""
    def __init__(self, io_handler: AgentIO):
        self.io = io_handler
        # Map action types to handler methods within this class
        self.ACTION_HANDLERS = {
            "think": self._handle_think,
            "message": self._handle_message,
            "run_command": self._handle_run_command,
        }

    def execute_action(self, action_type: str, action_element: ET.Element) -> bool:
        """Finds and executes the handler for a given action type."""
        handler = self.ACTION_HANDLERS.get(action_type)
        if handler:
            try:
                return handler(action_element) # Call the specific handler method
            except Exception as handler_e:
                self.io.display_error(f"Executing action '{action_type}': {handler_e}")
                return False # Indicate failure
        else:
            self.io.display_message(f"Agent: (Unknown action type: {action_type})")
            return False # Indicate no known action taken

    # --- Action Handler Methods ---

    def _handle_think(self, action: ET.Element) -> bool:
        """Handles the 'think' action."""
        content_element = action.find('content')
        content = content_element.text.strip() if content_element is not None and content_element.text else ""
        self.io.display_thought(content)
        return False # Does not count as an externally visible action

    def _handle_message(self, action: ET.Element) -> bool:
        """Handles the 'message' action."""
        content_element = action.find('content')
        content = content_element.text.strip() if content_element is not None and content_element.text else ""
        self.io.display_message(f"Agent: {content}")
        return True # Counts as an externally visible action

    def _handle_run_command(self, action: ET.Element) -> bool:
        """Handles the 'run_command' action (currently prints, doesn't execute)."""
        command_element = action.find('command_line')
        cwd_element = action.find('cwd')
        command_line = command_element.text.strip() if command_element is not None and command_element.text else None
        cwd = cwd_element.text.strip() if cwd_element is not None and cwd_element.text else None

        if command_line:
            self.io.display_command_request(command_line, cwd)
            # --- Placeholder for actual execution ---
            # (Actual execution logic would go here if enabled)
            # --- End Placeholder ---
            return True # Counts as an externally visible action
        else:
            self.io.display_message("Agent: (run_command action requested but no command_line found)")
            return False

# --- Simple Agent Class ---

class SimpleAgent:
    """A simple agent using simpledspy and XML-based action handling.

    Coordinates LLM calls, response parsing, and action execution.
    """

    XML_RESPONSE_SCHEMA = """\
<response>
  <actions>
    <!-- You can include zero or more of the following actions in any order -->
    <action type="think">
      <content>Your internal thought process or reasoning.</content>
    </action>
    <action type="message">
      <content>The message to display to the user.</content>
    </action>
    <action type="run_command">
      <command_line>The exact command line to execute.</command_line>
      <cwd>Optional: The working directory for the command.</cwd>
    </action>
  </actions>
</response>
"""

    def __init__(self, io_handler: AgentIO, action_executor: ActionExecutor, model_name: str):
        self.io = io_handler
        self.action_executor = action_executor # Use the provided executor
        self.model_name = model_name
        self.lm = None

    def _configure_lm(self) -> bool:
        """Configures the DSPy language model."""
        if self.lm:
            return True # Already configured
        try:
            self.lm = dspy.LM(self.model_name)
            dspy.settings.configure(lm=self.lm)
            self.io.display_message(f"DSPy LM configured successfully with {self.model_name}.")
            return True
        except Exception as e:
            self.io.display_error(f"Could not configure LM ({self.model_name}): {e}")
            self.io.display_error("Ensure API keys are set in environment.")
            self.lm = None
            return False

    def _get_llm_response(self, user_input: str) -> str:
        """Constructs prompt, calls LLM via simpledspy.pipe, returns raw response."""
        if not self.lm:
            self.io.display_error("LM not configured. Cannot get response.")
            return "<response><actions><action type='message'><content>Error: LM not configured.</content></action></actions></response>"

        prompt_text = f"""\
You are a helpful assistant. Respond to the user's request.
User request: "{user_input}"

Format your response strictly according to this XML schema:
{self.XML_RESPONSE_SCHEMA}

Include thoughts, messages to the user, and commands to run as needed within the <actions> tag.
"""
        self.io.display_message("Agent: Thinking...", end='\n') # Keep newline here
        try:
            raw_xml_response = pipe(
                prompt_text,
                description="Generate an XML response containing thoughts, messages, and commands based on the user request and schema."
            )
            self.io.display_raw_response(raw_xml_response)
            return raw_xml_response
        except Exception as e:
            self.io.display_error(f"During LLM call: {e}")
            return f"<response><actions><action type='message'><content>Error during LLM communication: {e}</content></action></actions></response>"

    def _parse_and_handle_actions(self, raw_xml_response: str):
        """Parses the XML response and delegates actions to the executor."""
        try:
            # Sanitize potential non-XML chars before parsing if needed, though ET should handle standard XML
            root = ET.fromstring(raw_xml_response)
            actions_element = root.find('actions')

            if actions_element is None:
                self.io.display_message("Agent: (No parsable <actions> tag found in response)")
                self.io.display_message(f"Agent: Received: {raw_xml_response}") # Show raw as fallback
                return

            has_executed_action = False
            for action in actions_element.findall('action'):
                action_type = action.get('type')
                # Delegate execution to the action_executor
                action_executed = self.action_executor.execute_action(action_type, action)
                if action_executed:
                    has_executed_action = True

            if not has_executed_action:
                # Use io handler
                self.io.display_message("Agent: (No message or command action executed)")

        except ET.ParseError as pe:
            self.io.display_error(f"Failed to parse XML response: {pe}")
            self.io.display_message(f"Agent: Received non-XML response:\n{raw_xml_response}") # Show raw
        except Exception as parse_e:
            self.io.display_error(f"Processing response: {parse_e}")

    def run(self):
        """Main interaction loop for the agent."""
        if not self._configure_lm():
            self.io.display_error("Agent cannot continue without a configured LM.")
            return # Exit if LM cannot be configured

        self.io.display_message(f"\nSimple DSPy Agent ({self.model_name}) - Type 'quit' to exit.")

        while True:
            try:
                user_input = self.io.get_input("User: ").strip()
                if user_input.lower() == 'quit':
                    break
                if not user_input:
                    continue

                raw_response = self._get_llm_response(user_input)
                self._parse_and_handle_actions(raw_response)

            except EOFError: # Handle Ctrl+D
                self.io.display_message("\nExiting due to EOF.")
                break
            except KeyboardInterrupt: # Handle Ctrl+C
                self.io.display_message("\nExiting due to keyboard interrupt.")
                break
            except Exception as loop_e:
                self.io.display_error(f"An unexpected error occurred in the main loop: {loop_e}")
                # Consider whether to break or continue after an unexpected loop error
                # break

        self.io.display_message("\nAgent session finished.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run a simple DSPy agent.")
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/google/gemini-2.0-flash-001",
        help="Name of the language model to use (e.g., 'deepseek/deepseek-chat')."
    )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # Instantiate components
    io_handler = ConsoleIO()
    action_executor = ActionExecutor(io_handler=io_handler)
    # Pass components to the agent
    agent = SimpleAgent(
        io_handler=io_handler,
        action_executor=action_executor,
        model_name=args.model
    )
    agent.run()