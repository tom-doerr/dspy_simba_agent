#!/usr/bin/env python3
import argparse
import sys
import coding_agent # Keep for version
# Import the refactored main function
from .main_logic import main

def main_cli():
    parser = argparse.ArgumentParser(prog="coding-agent")
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Run a dry run without dataset loading and optimization."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=coding_agent.__version__,
        help="Show program version and exit."
    )
    parser.add_argument(
        "--lm-name",
        dest="lm_name",
        default="deepseek/deepseek-chat",
        help="Name of the LM to use."
    )
    parser.add_argument(
        "--subset-size",
        dest="subset_size",
        type=int,
        default=32,
        help="Number of examples to use for optimization subset."
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=int,
        default=10,
        help="Timeout in seconds for functional evaluation."
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run mode: skipping dataset loading and optimization")
        sys.exit(0)

    # Call the main function from main_logic
    main(lm_name=args.lm_name, optimization_subset_size=args.subset_size, timeout=args.timeout)

if __name__ == "__main__":
    main_cli()
