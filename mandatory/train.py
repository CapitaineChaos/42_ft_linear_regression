#!/usr/bin/env python3
# Assisted-by: CLAUDE_SONNET:4.6_HIGH

# Usage : ./train.py ../data/data.csv

import argparse
import sys
import matplotlib.pyplot as plt


def read_dataset(dataset_path):
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
# In our case, it's a linear function of the form hθ​(x) = θ0 + θ1 * x, where θ0 is the intercept and θ1 is the slope.
# hθ​(x) = θ0 + θ1 * x
def estimate_price(km, theta0, theta1):
    return theta0 + theta1 * km


def update_thetas(dataset, learning_rate, theta0, theta1):
    # J stands for "cost function" or "loss function", and is a measure of how well the model fits the data.
    # J(θ0, θ1) = (1/2m) * Σ(hθ​(xᵢ) - yᵢ)², where hθ​(xᵢ) = θ0 + θ1 * xᵢ ('2' is for the derivative to be simpler)
    # ∇ stands for "gradient", and is a vector that points in the direction of the steepest increase of the cost function.
    # ∇J(θ0, θ1) = (∂J/∂θ0, ∂J/∂θ1)
    # ∂J/∂θ0 = (1/m) * Σ(hθ​(xᵢ) - yᵢ)
    # ∂J/∂θ1 = (1/m) * Σ(hθ​(xᵢ) - yᵢ) * xᵢ
    m = len(dataset)
    sum0, sum1 = 0.0, 0.0
    for km, price in dataset:
        # error = hθ​(xᵢ) - yᵢ = (θ0 + θ1 * xᵢ) - yᵢ
        # error is the difference between the predicted price and the actual price for each data point
        error = estimate_price(km, theta0, theta1) - price
        sum0 += error
        sum1 += error * km
    # tmpθ0 = η * ∂J/∂θ0
    tmp_theta0 = learning_rate / m * sum0
    # tmpθ1 = η * ∂J/∂θ1
    tmp_theta1 = learning_rate / m * sum1
    # θ0 := θ0 - η * ∂J/∂θ0
    theta0 -= tmp_theta0
    # θ1 := θ1 - η * ∂J/∂θ1
    theta1 -= tmp_theta1
    return theta0, theta1



def compute_mse(dataset, theta0, theta1):
    # MSE stands for "Mean Squared Error", and is a common metric for evaluating the performance of regression models.
    # MSE = (1 / m) * Σ(hθ​(xᵢ) - yᵢ)² = 2 * J(θ0, θ1)
    sum = 0.0
    for km, price in dataset:
        sum += (estimate_price(km, theta0, theta1) - price) ** 2
    return sum / len(dataset)


def init_plot():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.12, wspace=0.3)
    return fig, ax1, ax2


def update_plot(ax1, ax2, kms, prices, losses, theta0, theta1, epoch, max_epoch):
    current_loss = losses[-1] if losses else 0.0
    ax1.clear()
    ax2.clear()
    ax1.get_figure().suptitle(f'Epoch {epoch:,} / {max_epoch:,}', fontsize=13, fontweight='bold')
    ax1.scatter(kms, prices, alpha=0.6, label='Dataset')
    x_line = [min(kms), max(kms)]
    ax1.plot(x_line, [estimate_price(x, theta0, theta1) for x in x_line], 'r-', label=f'θ₀ = {theta0:.1f} θ₁ = {theta1:.6f}')
    ax1.set_xlabel('Mileage (km)')
    ax1.set_ylabel('Price')
    ax1.set_title('Linear Regression')
    ax1.legend()
    ax2.plot(losses, 'b-', label=f'MSE = {current_loss:.6f}')
    ax2.set_xlim(0, max_epoch)
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, max(losses) * 1.1 if losses else 1)
    ax2.set_ylabel('MSE')
    ax2.set_title('Loss Curve')
    ax2.legend()
    plt.pause(0.001)


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

def denormalize_theta(theta0, theta1, km_min, km_max, price_min, price_max):
    if km_max > km_min and price_max > price_min:
        theta1_denorm = theta1 * (price_max - price_min) / (km_max - km_min)
        theta0_denorm = price_min + theta0 * (price_max - price_min) - theta1_denorm * km_min
        return theta0_denorm, theta1_denorm
    else:
        return theta0, theta1

def train_model(dataset, plot=False):
    learning_rate = 0.1
    theta0, theta1 = 0.0, 0.0
    max_epoch = 1000
    update_every = 5
    kms = [d[0] for d in dataset]
    prices = [d[1] for d in dataset]
    losses = []

    dataset, km_min, km_max, price_min, price_max = normalize_dataset(dataset, kms, prices)
    if plot:
        fig, ax1, ax2 = init_plot()

    for epoch in range(max_epoch + 1):

        # Update the parameters θ0 and θ1 using gradient descent
        theta0, theta1 = update_thetas(dataset, learning_rate, theta0, theta1)


        if plot:
            losses.append(compute_mse(dataset, theta0, theta1))
            if epoch % update_every == 0 or epoch == max_epoch:
                if not plt.fignum_exists(fig.number):
                    print("\nWindow closed by user, stopping training...")
                    sys.exit(0)
                t0_plot, t1_plot = denormalize_theta(theta0, theta1, km_min, km_max, price_min, price_max)
                update_plot(ax1, ax2, kms, prices, losses, t0_plot, t1_plot, epoch, max_epoch)

    if plot:
        plt.ioff()
        plt.show()

    return denormalize_theta(theta0, theta1, km_min, km_max, price_min, price_max)


def save_thetas(theta0, theta1):
    with open("theta0", "w") as f:
        f.write(str(theta0))
    with open("theta1", "w") as f:
        f.write(str(theta1))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear regression model.")
    parser.add_argument("dataset_path", type=str, help="Path to the CSV dataset.")
    parser.add_argument("--plot", action="store_true", help="Display the regression in real-time.")
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = read_dataset(args.dataset_path)
    theta0, theta1 = train_model(dataset, plot=args.plot)
    print(f"theta0: {theta0}, theta1: {theta1}")
    save_thetas(theta0, theta1)

