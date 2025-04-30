#!/usr/bin/env python3
import argparse
from coding_agent import main

def main_cli():
    parser = argparse.ArgumentParser(prog="coding-agent")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a dry run without dataset loading and optimization."
    )
    args = parser.parse_args()
    if args.dry_run:
        print("Dry run mode: skipping dataset loading and optimization")
        return
    main()

if __name__ == "__main__":
    main_cli()
