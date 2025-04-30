import sys
from coding_agent.__main__ import main_cli
import coding_agent as ca

def test_cli_full_run(monkeypatch):
    # Stub heavy external calls
    monkeypatch.setattr(ca, "configure_lm", lambda name: None)
    monkeypatch.setattr(ca, "load_human_eval_dataset", lambda: [])
    # Simulate invocation with no flags
    monkeypatch.setattr(sys, "argv", ["coding-agent"])
    # Should exit cleanly
    main_cli()
