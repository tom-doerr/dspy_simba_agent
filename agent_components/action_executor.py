# agent_components/action_executor.py
import shlex
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional

from .io_handler import AgentIO # Import from the sibling module

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

    def execute_action(self, action_type: str, action_element: ET.Element) -> tuple[bool, str | None]:
        """Finds and executes the handler for a given action type.

        Returns:
            A tuple: (visible_action_executed: bool, command_output: Optional[str])
        """
        handler = self.ACTION_HANDLERS.get(action_type)
        if handler:
            try:
                return handler(action_element)  # Call the specific handler method
            except Exception as handler_e:
                self.io.display_error(f"Executing action '{action_type}': {handler_e}")
                return False, None  # Indicate failure
        else:
            self.io.display_message(f"Agent: (Unknown action type: {action_type})")
            return False, None  # Indicate no known action taken

    # --- Action Handler Methods ---

    def _handle_think(self, action: ET.Element) -> tuple[bool, str | None]:
        """Handles the 'think' action."""
        content_element = action.find("content")
        content = (
            content_element.text.strip()
            if content_element is not None and content_element.text
            else ""
        )
        self.io.display_thought(content)
        return False, None # Does not count as an externally visible action, no output

    def _handle_message(self, action: ET.Element) -> tuple[bool, str | None]:
        """Handles the 'message' action."""
        content_element = action.find("content")
        content = (
            content_element.text.strip()
            if content_element is not None and content_element.text
            else ""
        )
        self.io.display_message(f"Agent: {content}")
        return True, None # Counts as an externally visible action, no command output

    def _handle_run_command(self, action: ET.Element) -> tuple[bool, str | None]:
        """Handles the 'run_command' action and returns its output."""
        command_element = action.find("command_line")
        cwd_element = action.find("cwd")
        command_line = (
            command_element.text.strip()
            if command_element is not None and command_element.text
            else None
        )
        cwd = (
            cwd_element.text.strip()
            if cwd_element is not None and cwd_element.text
            else None
        )

        if command_line:
            self.io.display_command_request(command_line, cwd)
            # --- Actual execution --- 
            try:
                # Use shlex.split for safer command parsing
                command_parts = shlex.split(command_line)
                # Execute the command
                result = subprocess.run(
                    command_parts,
                    cwd=cwd, # Use the specified CWD or None if not provided
                    capture_output=True,
                    text=True,
                    check=False, # Don't raise exception on non-zero exit code
                    timeout=60 # Add a timeout for safety
                )
                command_output = f"--- Command: {command_line} ---\
Exit Code: {result.returncode}"
                if result.stdout:
                    command_output += f"\
--- stdout ---\
{result.stdout.strip()}"
                if result.stderr:
                    command_output += f"\
--- stderr ---\
{result.stderr.strip()}"
                command_output += "\
--- End Command Output ---"

                # Display output to user as well
                self.io.display_message("--- Command Output ---")
                if result.stdout:
                    self.io.display_message(result.stdout.strip())
                if result.stderr:
                    self.io.display_error(f"Command Error Output:\
{result.stderr.strip()}")
                if result.returncode != 0:
                    self.io.display_warning(f"Command finished with exit code: {result.returncode}") # Changed to warning
                self.io.display_message("--- End Command Output ---")

                return True, command_output # Return success and the combined output

            except FileNotFoundError:
                error_msg = f"Command not found: {shlex.split(command_line)[0]}"
                self.io.display_error(error_msg)
                return False, error_msg # Return failure and error message
            except subprocess.TimeoutExpired:
                error_msg = f"Command timed out: {command_line}"
                self.io.display_error(error_msg)
                return False, error_msg # Return failure and error message
            except Exception as e:
                error_msg = f"Failed to execute command '{command_line}': {e}"
                self.io.display_error(error_msg)
                return False, error_msg # Return failure and error message
             # --- End Execution (inside if command_line) ---
        else: # Corresponds to 'if command_line:'
            self.io.display_message(
                "Agent: (run_command action requested but no command_line found)"
            )
            return False, None # Indicate failure, no output
