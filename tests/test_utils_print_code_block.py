import coding_agent

def test_print_code_block(capsys):
    code = "foo()"
    coding_agent.print_code_block(code)
    captured = capsys.readouterr()
    expected = "```python\nfoo()\n```\n"
    assert captured.out == expected
