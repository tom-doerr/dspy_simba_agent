import pytest

# Adjust import based on new structure
from coding_agent.utils import print_code_block

def test_print_code_block(capsys):
    """Test printing a standard Python code block."""
    code = "foo()"
    print_code_block(code)
    captured = capsys.readouterr()
    # Check for line number and content, ignore exact formatting/styling
    assert " 1 " in captured.out # Check for line number 1
    assert code in captured.out # Check for the code itself

def test_print_code_block_with_language(capsys):
    """Test printing a code block with a specified language."""
    code = "<html></html>"
    print_code_block(code, language="html")
    captured = capsys.readouterr()
    # Check for line number and content
    assert " 1 " in captured.out # Check for line number 1
    assert code in captured.out # Check for the code itself

def test_print_code_block_empty(capsys):
    """Test printing an empty code block."""
    code = ""
    print_code_block(code)
    captured = capsys.readouterr()
    # For empty code, rich might just print the line number
    assert " 1 " in captured.out # Check for line number 1
    # Assert that the non-whitespace content is empty or minimal
    assert captured.out.strip() == "1" or captured.out.strip().startswith("1")
