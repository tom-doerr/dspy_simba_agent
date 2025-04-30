# DSPy SIMBA Coding Agent

[![CI](https://github.com/yourusername/yourrepo/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/yourrepo/actions/workflows/ci.yml)

## Overview

`dspy-simba-agent` is a Poetry-managed Python package and CLI tool that runs code synthesis on the HumanEval benchmark with DSPy SIMBA optimization.

## Installation

### Using Poetry

```bash
poetry install
```

### Using pip (editable)

```bash
pip install -e .
```

## Usage

```bash
coding-agent [--dry-run] [--version] [--help]
```

- `--dry-run`    Run without loading dataset or optimization.  
- `--version`    Show package version.  
- `--help`       Display help message.

## Development

Run the test suite:

```bash
poetry run pytest
```

## Contributing

1. Fork the repo on GitHub.  
2. Create a feature branch.  
3. Commit changes following conventional commits.  
4. Open a pull request.  
