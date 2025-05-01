import sys
from coding_agent.__main__ import main_cli
import coding_agent as ca

def test_cli_custom_args(monkeypatch):
    calls = []
    def fake_main(lm, subset, timeout):
        calls.append((lm, subset, timeout))
    monkeypatch.setattr(ca, "main", fake_main)
    monkeypatch.setattr(sys, "argv", [
        "coding-agent",
        "--lm-name", "modelX",
        "--subset-size", "5",
        "--timeout", "20"
    ])
    main_cli()
    assert calls == [("modelX", 5, 20)]
