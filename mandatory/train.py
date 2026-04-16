#!/usr/bin/env python3
# Assisted-by: CLAUDE_SONNET:4.6_HIGH

# Usage : ./train.py ../data/data.csv

import argparse
import sys


# Read the dataset into a dict where the key is 'km' and the value is 'price'
def read_dataset(dataset_path):
    dataset = []
    with open(dataset_path, 'r') as f:
        readline = f.readline()  # Skip the header
        if readline.strip() != "km,price":
            print("Warning: The first line of the dataset does not match the expected header 'km,price'.")
            exit(1)
        for line in f:
            try:
                km, price = line.strip().split(',')
                km = int(km)
                price = int(price)
            except ValueError:
                print(f"Warning: Skipping line with non-integer values: {line.strip()}")
                continue
            dataset.append((km, price))
    return dataset

def estimate_price(km, theta0, theta1):
    return theta0 + theta1 * km

def train_model(dataset):
    learning_rate = 1e-10
    theta0 = 0.0
    theta1 = 0.0
    len_ = len(dataset)
    max_epochs = 10000000
    for epoch in range(max_epochs):
        sum0 = 0.0
        sum1 = 0.0
        for km, price in dataset:
            error = estimate_price(km, theta0, theta1) - price
            sum0 += error
            sum1 += error * km
        theta0 -= learning_rate * (1 / len_) * sum0
        theta1 -= learning_rate * (1 / len_) * sum1
    return theta0, theta1

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

    # Train the model
    theta0, theta1 = train_model(dataset)
    print(f"theta0: {theta0}, theta1: {theta1}")

    # Save values to files theta0 and theta1
    with open("theta0", "w") as f:
        f.write(str(theta0))
    with open("theta1", "w") as f:
        f.write(str(theta1))

