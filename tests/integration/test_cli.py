import subprocess
import sys

def test_cli_dry_run():
    result = subprocess.run(
        [sys.executable, "-m", "coding_agent", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Dry run mode: skipping dataset loading and optimization" in result.stdout
