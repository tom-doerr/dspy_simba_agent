# simpledspy_agent.py
import argparse
import xml.etree.ElementTree as ET
import dspy
from rich.console import Console
from rich.prompt import Prompt
from simpledspy import pipe

# Import the refactored components
from agent_components import ActionExecutor, AgentIO, ConsoleIO

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

    def __init__(
        self, io_handler: AgentIO, action_executor: ActionExecutor, model_name: str
    ):
        self.io = io_handler
        self.action_executor = action_executor  # Use the provided executor
        self.model_name = model_name
        self.lm = None

    def _configure_lm(self) -> bool:
        """Configures the DSPy language model."""
        if self.lm:
            return True  # Already configured
        try:
            self.lm = dspy.LM(self.model_name)
            dspy.settings.configure(lm=self.lm)
            self.io.display_message(
                f"DSPy LM configured successfully with {self.model_name}."
            )
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
        self.io.display_message("Agent: Thinking...", end="\n")  # Keep newline here
        try:
            raw_xml_response = pipe(
                prompt_text,
                description="Generate an XML response containing thoughts, messages, and commands based on the user request and schema.",
            )
            self.io.display_raw_response(raw_xml_response)
            return raw_xml_response
        except Exception as e:
            self.io.display_error(f"During LLM call: {e}")
            return f"<response><actions><action type='message'><content>Error during LLM communication: {e}</content></action></actions></response>"

    def _parse_and_handle_actions(
        self, raw_response: str
    ) -> tuple[bool, str | None]:
        """Parses the LLM's XML response and executes the contained actions.

        Returns:
            A tuple: (visible_action_occurred: bool, command_output: Optional[str])
        """
        visible_action_occurred = False
        command_output_from_action: str | None = None # Store output from a command

        # Attempt to clean the response (remove potential markdown fences)
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```xml"):
            cleaned_response = cleaned_response[len("```xml"):]
        if cleaned_response.startswith("```"):
             cleaned_response = cleaned_response[len("```"):]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-len("```")]
        cleaned_response = cleaned_response.strip() # Remove any extra whitespace

        # Handle potential fragments or missing root elements (optional refinement)
        if not cleaned_response:
            self.io.display_message("Received empty response after cleaning.")
            return False, None

        # Ensure the response ends correctly for parsing (existing logic adjusted)
        # This might need review depending on how robust fragment handling should be
        if not cleaned_response.strip().endswith("</response>"):
            if "<message>" in cleaned_response and "</message>" not in cleaned_response:
                cleaned_response += "</message>"
            if "<response>" in cleaned_response:
                 # Avoid adding a second response tag if one exists but isn't at start
                 if not cleaned_response.strip().startswith("<response>"):
                     cleaned_response += "</response>"
            elif "<" in cleaned_response and ">" in cleaned_response:
                 # If tags exist but no <response>, wrap it
                 cleaned_response = f"<response>{cleaned_response}</response>"
            else:
                 # Treat as plain text if no tags suggest XML structure
                 self.io.display_message(f"Agent: {cleaned_response}")
                 return False, None

        if not cleaned_response.strip().startswith("<response>"):
             # If still doesn't start with response, check again
             if "<" in cleaned_response and ">" in cleaned_response:
                  cleaned_response = f"<response>{cleaned_response}</response>"
             else:
                  self.io.display_message(f"Agent: {cleaned_response}")
                  return False, None

        try:
            # Parse the cleaned XML string
            root = ET.fromstring(cleaned_response)

            actions_element = root.find("actions")
            if actions_element is None:
                # Maybe the response *is* the action list?
                actions_element = root if root.tag == 'actions' else None 
                # Or maybe it's a simple message fallback?
                if actions_element is None:
                    message_element = root.find("message") # Legacy format?
                    if message_element is not None and message_element.text:
                        self.io.display_message(f"Agent: {message_element.text.strip()}")
                        return False, None # No command output here
                    else:
                         self.io.display_message("No <actions> tag found in response.")
                         self.io.display_raw_response(cleaned_response)
                         return False, None # No command output here

            for action in actions_element:
                action_type = action.get("type")
                if not action_type:
                    self.io.display_warning("Action tag found without a 'type' attribute.")
                    continue

                # Delegate execution to the action_executor
                action_executed, current_command_output = self.action_executor.execute_action(
                    action_type, action
                )
                if action_executed:
                    visible_action_occurred = True
                    # If a command was run and produced output, store it
                    if action_type == "run_command" and current_command_output:
                        command_output_from_action = current_command_output
                        # For now, only handle output from the *last* run_command action if multiple exist

        except ET.ParseError as e:
            self.io.display_error(f"Failed to parse XML response: {e}")
            # If parsing fails, we likely didn't get command output

            # Display the problematic response for debugging
            self.io.display_message(f"Received non-XML or malformed response:\n{raw_response}") # Use raw_response here
            return False, None # Return failure, no command output

        except Exception as parse_e:
            # Catch any other unexpected errors during parsing/handling
            self.io.display_error(f"Error processing response: {parse_e}")
            self.io.display_message(f"Raw Response:\n{raw_response}")
            return False, None # Return failure, no command output

        # Return the results after the try block completes successfully
        return visible_action_occurred, command_output_from_action

    def run(self):
        """Main loop for the agent interaction."""
        if not self._configure_lm():
            self.io.display_error("Agent cannot continue without a configured LM.")
            return  # Exit if LM cannot be configured

        self.io.display_message(
            f"\nSimple DSPy Agent ({self.model_name}) - Type 'quit' to exit."
        )

        while True:
            try:
                user_input = self.io.get_input("User: ").strip()
                if user_input.lower() == "quit":
                    break
                if not user_input:
                    continue

                # --- First LLM Call --- 
                raw_response = self._get_llm_response(user_input)

                # --- First Action Execution --- 
                visible_action_occurred_first_pass, command_output = self._parse_and_handle_actions(raw_response)

                # --- Second LLM Call (if command ran and returned output) --- 
                if command_output:
                    self.io.display_thought(
                        "Command executed and returned output. Asking LLM for final response based on output..."
                    )
                    # Construct a new prompt combining original input and command result
                    follow_up_prompt = (
                        f"Original user request: {user_input}\n\n"
                        f"I previously ran a command and got the following output:\n"
                        f"{command_output}\n\n"
                        f"Based on the original request and this output, what is the final response or next action?"
                    )
                    # Get the second response
                    second_raw_response = self._get_llm_response(follow_up_prompt)

                    # --- Second Action Execution --- 
                    # Execute actions from the second response. We typically expect a 'message' action here.
                    _visible_action_occurred_second_pass, _ = self._parse_and_handle_actions(
                        second_raw_response
                    )
                # else: No command output, the first pass was sufficient.

            except EOFError:  # Handle Ctrl+D
                self.io.display_message("\nExiting due to EOF.")
                break
            except KeyboardInterrupt:  # Handle Ctrl+C
                self.io.display_message("\nExiting due to keyboard interrupt.")
                break
            except Exception as loop_e:
                self.io.display_error(
                    f"An unexpected error occurred in the main loop: {loop_e}"
                )
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
        help="Name of the language model to use (e.g., 'deepseek/deepseek-chat').",
    )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # Instantiate components
    io_handler = ConsoleIO()
    action_executor = ActionExecutor(io_handler=io_handler)
    # Pass components to the agent
    agent = SimpleAgent(
        io_handler=io_handler, action_executor=action_executor, model_name=args.model
    )
    agent.run()
