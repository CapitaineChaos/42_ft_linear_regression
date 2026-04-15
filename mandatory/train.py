#!/usr/bin/env python3
# Assisted-by: CLAUDE_SONNET:4.6_HIGH

# Usage : ./train.py ../data/data.csv

import argparse
import sys

# Read the dataset into a dict where the key is 'km' and the value is 'price'
def read_dataset(dataset_path):
    dataset = {}
    with open(dataset_path, 'r') as f:
        readline = f.readline()  # Skip the header
        if readline.strip() != "km,price":
            print("Warning: The first line of the dataset does not match the expected header 'km,price'.")
            exit(1)
        for line in f:
            km, price = line.strip().split(',')
            try:
                km = int(km)
                price = int(price)
            except ValueError:
                print(f"Warning: Skipping line with non-integer values: {line.strip()}")
                continue
            dataset[int(km)] = int(price)
    return dataset

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

    # Read the dataset
    dataset = read_dataset(dataset_path)
    print(dataset)

