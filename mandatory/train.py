#!/usr/bin/env python3
# Assisted-by: CLAUDE_SONNET:4.6_HIGH / GPT:5.4_THINKING

# Usage : ./train.py ../data/data.csv

import argparse
import sys
import os
import signal

def handler(sig, frame):
    print("\nInterrupt received, stopping training...")
    sys.exit(0)


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


# h stands for "hypothesis", and is the function we want to fit to the data.
# In our case, it's a linear function of the form hθ​(x) = θ₀ + θ₁ * x, where θ₀ is the intercept and θ₁ is the slope.
# hθ​(x) = θ₀ + θ₁ * x
def estimate_price(km, theta0, theta1):
    return theta0 + theta1 * km


def update_thetas(dataset, learning_rate, theta0, theta1):
    # J stands for "cost function" or "loss function", and is a measure of how well the model fits the data.
    # J(θ₀, θ₁) = (1/2m) * Σ(hθ​(xᵢ) - yᵢ)², where hθ​(xᵢ) = θ₀ + θ₁ * xᵢ ('2' is for the derivative to be simpler)
    # ∇ stands for "gradient", and is a vector that points in the direction of the steepest increase of the cost function.
    # ∇J(θ₀, θ₁) = (∂J/∂θ₀, ∂J/∂θ₁)
    # ∂J/∂θ₀ = (1/m) * Σ(hθ​(xᵢ) - yᵢ)
    # ∂J/∂θ₁ = (1/m) * Σ(hθ​(xᵢ) - yᵢ) * xᵢ
    m = len(dataset)
    sum0, sum1 = 0.0, 0.0
    tmp_theta0, tmp_theta1 = 0.0, 0.0
    for km, price in dataset:
        # error = hθ​(xᵢ) - yᵢ = (θ₀ + θ₁ * xᵢ) - yᵢ
        # error is the difference between the predicted price and the actual price for each data point
        error = estimate_price(km, theta0, theta1) - price
        sum0 += error
        sum1 += error * km
    # tmpθ₀ = η * ∂J/∂θ₀
    tmp_theta0 = learning_rate / m * sum0
    # tmpθ₁ = η * ∂J/∂θ₁
    tmp_theta1 = learning_rate / m * sum1
    return tmp_theta0, tmp_theta1


# Min max normalization
def normalize_dataset(dataset, kms, prices):
    km_min, km_max = min(kms), max(kms)
    price_min, price_max = min(prices), max(prices)
    normalized = []
    for km, price in dataset:
        norm_km = (km - km_min) / (km_max - km_min) if km_max > km_min else 0.0
        norm_price = (price - price_min) / (price_max - price_min) if price_max > price_min else 0.0
        normalized.append((norm_km, norm_price))
    return normalized, km_min, km_max, price_min, price_max


def denormalize_thetas(theta0, theta1, km_min, km_max, price_min, price_max):
    if km_max > km_min and price_max > price_min:
        theta1_denorm = theta1 * (price_max - price_min) / (km_max - km_min)
        theta0_denorm = price_min + theta0 * (price_max - price_min) - theta1_denorm * km_min
        return theta0_denorm, theta1_denorm
    else:
        return theta0, theta1


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


def train_model(dataset):
    learning_rate = 0.1
    theta0, theta1 = 0.0, 0.0
    nb_epochs = 1000

    kms = [d[0] for d in dataset]
    prices = [d[1] for d in dataset]

    dataset, km_min, km_max, price_min, price_max = normalize_dataset(dataset, kms, prices)

    for _ in range(1, nb_epochs + 1):

        # Update the parameters θ₀ and θ₁ using gradient descent
        tmp_theta0, tmp_theta1 = update_thetas(dataset, learning_rate, theta0, theta1)
        # θ₀ := θ₀ - η * ∂J/∂θ₀
        theta0 -= tmp_theta0
        # θ₁ := θ₁ - η * ∂J/∂θ₁
        theta1 -= tmp_theta1

    return denormalize_thetas(theta0, theta1, km_min, km_max, price_min, price_max)


def save_thetas(theta0, theta1):
    try:
        with open("theta0", "w") as f:
            f.write(str(theta0))
        with open("theta1", "w") as f:
            f.write(str(theta1))
    except IOError as e:
        print(f"Error: could not save thetas to files: {e}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear regression model.")
    parser.add_argument("dataset_path", type=str, help="Path to the CSV dataset.")
    if len(sys.argv) == 1 or len(sys.argv) > 2:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    signal.signal(signal.SIGINT, handler)
    dataset = read_dataset(args.dataset_path)
    theta0, theta1 = train_model(dataset)
    print(f"θ₀: {theta0}, θ₁: {theta1}")
    save_thetas(theta0, theta1)

