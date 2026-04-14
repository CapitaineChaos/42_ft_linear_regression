#!/usr/bin/env python3
# Assisted-by: CLAUDE_SONNET:4.6_HIGH

import argparse
import sys

# Main
if __name__ == "__main__":
    # Reading arguments
    parser = argparse.ArgumentParser(description="Train a model on the given dataset.")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset to be used for training.",
    )
    
    # If no arguments are provided or more than 1, print help and exit
    if len(sys.argv) == 1 or len(sys.argv) > 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    dataset_path = args.dataset_path

