#!/usr/bin/env python3
# Assisted-by: CLAUDE_SONNET:4.6_HIGH

# Usage : ./train.py ../data/data.csv

import argparse
import sys
import math
import os
import matplotlib.pyplot as plt


def init_plot():
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.08, wspace=0.35, hspace=0.4)
    return fig, ax1, ax2, ax3, ax4


def precompute_cost_grid(norm_dataset, n=70):
    t0_vals = [-0.5 + 2.0 * j / (n - 1) for j in range(n)]  # linspace(-0.5, 1.5)
    t1_vals = [-2.0 + 3.0 * i / (n - 1) for i in range(n)]  # linspace(-2.0, 1.0)
    T0 = [[t0_vals[j] for j in range(n)] for i in range(n)]
    T1 = [[t1_vals[i] for j in range(n)] for i in range(n)]
    J = [[math.log(compute_cost(norm_dataset, T0[i][j], T1[i][j]) + 1e-10)
          for j in range(n)] for i in range(n)]
    return T0, T1, J


def update_plot(ax1, ax2, ax3, ax4, kms, prices, losses, r2_history, theta0, theta1, epoch, nb_epochs, history, cost_grid):
    current_loss = losses[-1] if losses else 0.0
    current_r2 = r2_history[-1] if r2_history else 0.0
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    ax1.get_figure().suptitle(f'Epoch {epoch:,} / {nb_epochs:,}', fontsize=13, fontweight='bold')
    ax1.scatter(kms, prices, alpha=0.6, label='Dataset')
    x_line = [min(kms), max(kms)]
    ax1.plot(x_line, [estimate_price(x, theta0, theta1) for x in x_line], 'r-', label=f'θ₀ = {theta0:.1f} θ₁ = {theta1:.6f}')
    ax1.set_xlabel('Mileage (km)')
    ax1.set_ylabel('Price')
    ax1.set_title('Linear Regression')
    ax1.legend()

    ax2.plot(losses, 'b-', label=f'MSE = {current_loss:.6f}')
    ax2.set_xlim(0, nb_epochs)
    upper = max(losses) * 1.1 if losses and max(losses) > 0 else 1
    ax2.set_ylim(0, upper)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Loss Curve')
    ax2.legend()

    T0, T1, J_grid = cost_grid
    h_t0 = [h[0] for h in history]
    h_t1 = [h[1] for h in history]
    ax3.contour(T0, T1, J_grid, levels=20, cmap='plasma', linewidths=0.8)
    ax3.plot(h_t0, h_t1, 'r.-', markersize=2, label='Gradient descent path')
    ax3.scatter(h_t0[0], h_t1[0], c='blue', s=50, zorder=5, label='Start')
    ax3.scatter(h_t0[-1], h_t1[-1], c='red', s=50, zorder=5, label='Current')
    ax3.set_xlabel('θ₀ (normalised)')
    ax3.set_ylabel('θ₁ (normalised)')
    ax3.set_title('Parameters space (J isolines)')
    ax3.legend(fontsize=7)

    ax4.plot(r2_history, 'g-', label=f'R² = {current_r2:.6f}')
    ax4.set_xlim(0, nb_epochs)
    # No needs to see negative R² values, and R² > 1 is not possible
    ax4.set_ylim(0, 1.05)
    ax4.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('R²')
    ax4.set_title('Coefficient of determination R²')
    ax4.legend()

    plt.pause(0.01)


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
    tmp_theta0, tmp_theta1 = 0.0, 0.0
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
    return tmp_theta0, tmp_theta1


def compute_mse(dataset, theta0, theta1):
    # MSE stands for "Mean Squared Error", and is a common metric for evaluating the performance of regression models.
    # MSE = (1 / m) * Σ(hθ​(xᵢ) - yᵢ)² = 2 * J(θ0, θ1)
    sum = 0.0
    for km, price in dataset:
        sum += (estimate_price(km, theta0, theta1) - price) ** 2
    return sum / len(dataset)


def compute_cost(dataset, theta0, theta1):
    return compute_mse(dataset, theta0, theta1) / 2

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
    # Degenerate case: all prices are identical → constant model, slope is 0
    if price_max == price_min:
        return price_min, 0.0
    # Degenerate case: all kms are identical → slope not identifiable, intercept is mean price
    if km_max == km_min:
        return (price_min + price_max) / 2.0, 0.0
    theta1_denorm = theta1 * (price_max - price_min) / (km_max - km_min)
    theta0_denorm = price_min + theta0 * (price_max - price_min) - theta1_denorm * km_min
    return theta0_denorm, theta1_denorm


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
    history = [(theta0, theta1)]
    nb_epochs = 1000
    update_every = 10
    kms = [d[0] for d in dataset]
    prices = [d[1] for d in dataset]
    losses = []
    r2_history = []

    dataset, km_min, km_max, price_min, price_max = normalize_dataset(dataset, kms, prices)

    cost_grid = precompute_cost_grid(dataset)
    fig, ax1, ax2, ax3, ax4 = init_plot()

    for epoch in range(1, nb_epochs + 1):

        # Update the parameters θ0 and θ1 using gradient descent
        tmp_theta0, tmp_theta1 = update_thetas(dataset, learning_rate, theta0, theta1)

        # θ0 := θ0 - η * ∂J/∂θ0
        theta0 -= tmp_theta0
        # θ1 := θ1 - η * ∂J/∂θ1
        theta1 -= tmp_theta1


        # Save the history of θ0 and θ1 for analysis or plotting after training
        history.append((theta0, theta1))
        # Save the current loss history for plotting
        losses.append(compute_mse(dataset, theta0, theta1))
        t0_plot, t1_plot = denormalize_thetas(theta0, theta1, km_min, km_max, price_min, price_max)
        real_dataset = list(zip(kms, prices))
        # Save the current R² score history for plotting
        r2_history.append(calculate_r2_score(real_dataset, t0_plot, t1_plot))
        if epoch % update_every == 0 or epoch == nb_epochs:
            if not plt.fignum_exists(fig.number):
                print("\nWindow closed by user, stopping training...")
                sys.exit(0)
            update_plot(ax1, ax2, ax3, ax4, kms, prices, losses, r2_history, t0_plot, t1_plot, epoch, nb_epochs, history, cost_grid)

    # After training is complete, keep the final plot open until the user closes it
    plt.ioff()
    plt.show()

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
    dataset = read_dataset(args.dataset_path)
    theta0, theta1 = train_model(dataset)
    print(f"theta0: {theta0}, theta1: {theta1}")
    save_thetas(theta0, theta1)

