# Utility functions

def print_code_block(code: str) -> None:
    """Prints a string formatted as a Python code block in Markdown."""
    print("```python")
    print(code)
    print("```")
