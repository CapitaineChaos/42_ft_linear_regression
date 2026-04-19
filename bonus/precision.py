#!/usr/bin/env python3
# Assisted-by: CLAUDE_SONNET:4.6_HIGH / GPT:5.4_THINKING

# Usage : ./precision.py ../data/data.csv


import argparse
import sys
import os


def estimate_price(km, theta0, theta1):
    return theta0 + theta1 * km


# Calculate R² score
def calculate_r2_score(dataset, theta0, theta1):
    y_true = [price for _, price in dataset]
    y_pred = [estimate_price(km, theta0, theta1) for km, _ in dataset]

    ss_res = 0.0
    ss_tot = 0.0
    for yt, yp in zip(y_true, y_pred):
        ss_res += (yt - yp) ** 2
    yt_sum = 0.0
    for yt in y_true:
        yt_sum += yt
    mean_y = yt_sum / len(y_true)
    for yt in y_true:
        ss_tot += (yt - mean_y) ** 2
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


def read_theta(filename):
    if not os.path.isfile(filename) or not os.access(filename, os.R_OK):
        return None
    with open(filename, 'r') as f:
        return float(f.read().strip())


def read_dataset(dataset_path):
    # If file not exists or is not readable, print an error message and exit
    if not os.path.isfile(dataset_path) or not os.access(dataset_path, os.R_OK):
        print(f"Error: file '{dataset_path}' does not exist or is not readable.")
        sys.exit(1)
    dataset = []
    with open(dataset_path, 'r') as f:
        if f.readline().strip() != "km,price":
            print("Error: unexpected header, expected 'km,price'.")
            sys.exit(1)
        for line in f:
            try:
                km, price = line.strip().split(',')
                dataset.append((int(km), int(price)))
            except ValueError:
                print(f"Warning: skipping invalid line: {line.strip()}")
    if not dataset:
        print("Error: no valid data found in the dataset.")
        sys.exit(1)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the precision of the trained model on the dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the CSV dataset.")
    if len(sys.argv) == 1 or len(sys.argv) > 2:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = read_dataset(args.dataset_path)

    # Read theta0 and theta1 from files
    theta0 = read_theta("theta0")
    theta1 = read_theta("theta1")

    if theta0 is None or theta1 is None:
        print("Error: theta0 and theta1 must be available in the current directory.")
        sys.exit(1)
        
    r2_score = calculate_r2_score(dataset, theta0, theta1)
    print(f"R² score of the model on the dataset: {r2_score:.4f}")