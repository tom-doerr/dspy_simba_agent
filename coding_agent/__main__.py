#!/usr/bin/env python3
import argparse
from coding_agent import main, __version__

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
        version=__version__,
        help="Show program version and exit."
    )
    args = parser.parse_args()
    if args.dry_run:
        print("Dry run mode: skipping dataset loading and optimization")
        return
    main()

if __name__ == "__main__":
    main_cli()
