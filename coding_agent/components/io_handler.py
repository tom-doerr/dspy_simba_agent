# agent_components/io_handler.py
from abc import ABC, abstractmethod
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

# --- IO Handling Abstraction and Implementation ---

class AgentIO(ABC):
    """Abstract base class for handling agent input and output."""

    @abstractmethod
    def get_input(self, prompt: str) -> str:
        """Get input from the user."""
        pass

    @abstractmethod
    def display_message(self, message: str):
        """Display a regular message from the agent."""
        pass

    @abstractmethod
    def display_thought(self, thought: str):
        """Display the agent's internal thought process."""
        pass

    @abstractmethod
    def display_command_request(self, command: str, cwd: str | None):
        """Display a command the agent wants to run."""
        pass

    @abstractmethod
    def display_error(self, error_message: str):
        """Display an error message."""
        pass

    @abstractmethod
    def display_warning(self, warning_message: str):
        """Display a warning message."""
        pass

    @abstractmethod
    def display_raw_response(self, raw_response: str):
        """Display the raw, unparsed response from the LLM."""
        pass


class ConsoleIO(AgentIO):
    """Console implementation of AgentIO using rich for formatted output."""

    def __init__(self):
        self.console = Console()

    def get_input(self, prompt: str) -> str:
        """Gets input from the console."""
        return self.console.input(f"[bold cyan]{prompt}[/bold cyan]")

    def display_message(self, message: str):
        """Displays a message to the console."""
        self.console.print(message)

    def display_thought(self, thought: str):
        """Displays the agent's thought process in a panel."""
        self.console.print(Panel(thought, title="Agent (Thought)", style="dim cyan"))

    def display_command_request(self, command: str, cwd: str | None):
        """Displays the command the agent wants to run."""
        cwd_info = f" in directory '{cwd}'" if cwd else ""
        self.console.print(f"[bold yellow]Agent wants to run command{cwd_info}:[/bold yellow]")
        # Use rich.syntax for potentially better highlighting if command is complex
        syntax = Syntax(command, "bash", theme="default", line_numbers=False)
        self.console.print(syntax)
        # For safety, indicate execution status
        # self.console.print("  (Command execution is currently enabled)")

    def display_error(self, error_message: str):
        """Displays an error message in red."""
        self.console.print(f"[bold red]Error:[/bold red] {error_message}")

    def display_warning(self, warning_message: str):
        """Displays a warning message in yellow."""
        self.console.print(f"[bold yellow]Warning:[/bold yellow] {warning_message}")

    def display_raw_response(self, raw_response: str):
        """Displays the raw LLM response in a panel for debugging."""
        self.console.print(Panel(raw_response, title="--- Raw XML Response ---", border_style="grey50"))
        self.console.print("-" * 24) # Separator
