# Utility functions

import sys
from rich.console import Console 
from rich.syntax import Syntax   

CONSOLE = Console() 

def print_code_block(code: str, language: str = "python") -> None:
    """Prints a code block using rich for syntax highlighting."""
    # CONSOLE.print(f"--- Code Block ({language}) ---")
    syntax = Syntax(code, language, theme="default", line_numbers=True)
    CONSOLE.print(syntax)
    # CONSOLE.print("--- End Code Block ---")
