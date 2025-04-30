import subprocess
import sys
import coding_agent

def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "coding_agent", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert coding_agent.__version__ in result.stdout
