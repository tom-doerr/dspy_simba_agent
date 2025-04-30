import pytest
import re
import coding_agent
try:
    # Python ≥3.8
    from importlib.metadata import version as pkg_version, PackageNotFoundError
except ImportError:
    # Older Python
    from importlib_metadata import version as pkg_version, PackageNotFoundError

def test_version_format():
    v = coding_agent.__version__
    # Simple SemVer check: at least major.minor.patch with numeric parts
    assert re.match(r'^\d+\.\d+\.\d+(\.\d+)?$', v), f"Version '{v}' is not SemVer"

def test_version_matches_metadata():
    try:
        meta_v = pkg_version("dspy-simba-agent")
    except PackageNotFoundError:
        pytest.skip("Package not installed; skipping metadata version check")
    assert coding_agent.__version__ == meta_v
