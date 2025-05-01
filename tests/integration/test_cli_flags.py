import sys
# Import the module containing the CLI entrypoint
import coding_agent.__main__

def test_cli_custom_args(monkeypatch):
    calls = []
    # Define the fake main function with the correct signature
    # (matching the signature of coding_agent.main_logic.main)
    def fake_main(lm_name: str, optimization_subset_size: int, timeout: int):
        calls.append((lm_name, optimization_subset_size, timeout))
    # Patch the 'main' function *within the __main__ module's namespace*
    monkeypatch.setattr(coding_agent.__main__, "main", fake_main)
    # Simulate command line arguments
    monkeypatch.setattr(sys, "argv", [
        "coding-agent",
        "--lm-name", "modelX",
        "--subset-size", "5",
        "--timeout", "20"
    ])
    # Run the CLI entry point
    coding_agent.__main__.main_cli()
    # Assert that the fake main was called with the correct arguments
    assert calls == [("modelX", 5, 20)]
