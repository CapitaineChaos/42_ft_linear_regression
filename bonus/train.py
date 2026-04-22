#!/usr/bin/env python3
# Assisted-by: CLAUDE_SONNET:4.6_HIGH / GPT:5.4_THINKING

# Usage : ./train.py ../data/data.csv

import argparse
import sys
import math
import os
import signal
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def handler(sig, frame):
    print("\nInterrupt received, stopping training...")
    sys.exit(0)


def init_plot(title):
    # Enable the interactive mode in Matplotlib
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    fig.canvas.manager.set_window_title(title)

    # Create 4 subplots:
    # (cols, rows, index) 
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.05, wspace=0.18, hspace=0.28)
    return fig, ax1, ax2, ax3, ax4


def precompute_cost_grid(norm_dataset, n=70):
    # Range of θ₀: [-0.5, 1.5]
    t0_vals = [-0.5 + 2.0 * j / (n - 1) for j in range(n)]
    # Range of θ₁: [-2.0, 1.0]
    t1_vals = [-2.0 + 3.0 * i / (n - 1) for i in range(n)]
    # Create a grid of θ₀
    T0 = [[t0_vals[j] for j in range(n)] for _ in range(n)]
    # Create a grid of θ₁
    T1 = [[t1_vals[i] for _ in range(n)] for i in range(n)]
    # Compute the cost J for each pair (θ₀, θ₁) in the grid
    # Take the logarithm to reduce the range of values for better visualization
    J = [[math.log(compute_cost(norm_dataset, T0[i][j], T1[i][j]) + 1e-10)
          for j in range(n)] for i in range(n)]
    return T0, T1, J


def update_plot1(ax1, kms, prices, theta0, theta1, epoch, nb_epochs):
    ax1.clear()
    ax1.get_figure().suptitle(f'Epoch {epoch:,} / {nb_epochs:,}', fontsize=13, fontweight='bold')
    for km, price in zip(kms, prices):
        predicted = estimate_price(km, theta0, theta1)
        color = 'green' if price >= predicted else 'red'
        ax1.plot([km, km], [price, predicted], color=color, linewidth=0.8, alpha=0.6)
    ax1.scatter(kms, prices, alpha=0.6, zorder=3, label='Dataset')
    x_line = [min(kms), max(kms)]
    ax1.plot(x_line, [estimate_price(x, theta0, theta1) for x in x_line], 'b-', label='ŷ = θ₀ + θ₁·x')
    ax1.plot([], [], ' ', label=f'θ₀ = {theta0:.4f}')
    ax1.plot([], [], ' ', label=f'θ₁ = {theta1:.6f}')
    # ME stands for "Mean Error", and is the average of the differences between the predicted prices and the actual prices.
    # ME = (1 / m) * Σ(hθ​(xᵢ) - ŷᵢ)
    me = 0.0
    for km, price in zip(kms, prices):
        me += price - estimate_price(km, theta0, theta1)
    me /= len(kms)
    ax1.set_xlabel('Mileage (km)')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Linear Regression  -  ME = {me:.2f}')
    ax1.legend(fontsize=7)


def update_plot2(ax2, losses, nb_epochs):
    current_loss = losses[-1] if losses else 0.0
    ax2.clear()
    ax2.plot(losses, 'b-', label=f'MSE = {current_loss:.6f}')
    ax2.set_xlim(0, nb_epochs)
    upper = max(losses) * 1.1 if losses and max(losses) > 0 else 1
    ax2.set_ylim(0, upper)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Loss Curve')
    ax2.legend(fontsize=7)


def update_plot3(ax3, history, cost_grid):
    ax3.clear()
    T0, T1, J_grid = cost_grid
    h_t0 = [h[0] for h in history]
    h_t1 = [h[1] for h in history]
    ax3.contour(T0, T1, J_grid, levels=20, cmap='plasma', linewidths=0.5)
    ax3.plot(h_t0, h_t1, 'r.-', markersize=1, label='Gradient descent path')
    ax3.scatter(h_t0[0], h_t1[0], c='blue', s=50, zorder=5, label='Start')
    ax3.scatter(h_t0[-1], h_t1[-1], c='red', s=50, zorder=5, label='Current')
    ax3.set_xlabel('θ₀ (normalised)')
    ax3.set_ylabel('θ₁ (normalised)')
    ax3.set_title('Contour plot gradient descent path')
    ax3.legend(fontsize=7)


def update_plot4(ax4, r2_history, nb_epochs):
    ax4.clear()
    current_r2 = r2_history[-1] if r2_history else 0.0
    ax4.plot(r2_history, 'g-', label=f'R² = {current_r2:.6f}')
    ax4.set_xlim(0, nb_epochs)
    # No needs to see negative R² values, and R² > 1 is not possible
    ax4.set_ylim(0, 1.05)
    ax4.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('R²')
    ax4.set_title('Coefficient of determination R²')
    ax4.legend(fontsize=7)


def update_plot(ax1, ax2, ax3, ax4, kms, prices, losses, r2_history, theta0, theta1, epoch, nb_epochs, history, cost_grid):
    update_plot1(ax1, kms, prices, theta0, theta1, epoch, nb_epochs)
    update_plot2(ax2, losses, nb_epochs)
    update_plot3(ax3, history, cost_grid)
    update_plot4(ax4, r2_history, nb_epochs)
    plt.pause(0.005)


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


def compute_mse(dataset, theta0, theta1):
    # MSE stands for "Mean Squared Error", and is a common metric for evaluating the performance of regression models.
    # MSE = (1 / m) * Σ(hθ​(xᵢ) - yᵢ)² = 2 * J(θ₀, θ₁)
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
    # Degenerate case: all prices are identical -> constant model, slope is 0
    if price_max == price_min:
        return price_min, 0.0
    # Degenerate case: all kms are identical -> slope not identifiable, intercept is mean price
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
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)


def train_model(dataset, dataset_path):
    learning_rate = 0.15
    theta0, theta1 = 0.0, 0.0
    history = [(theta0, theta1)]
    nb_epochs = 1200
    update_every = 20
    kms = [d[0] for d in dataset]
    prices = [d[1] for d in dataset]
    losses = []
    r2_history = []

    dataset, km_min, km_max, price_min, price_max = normalize_dataset(dataset, kms, prices)

    cost_grid = precompute_cost_grid(dataset)
    fig, ax1, ax2, ax3, ax4 = init_plot(os.path.basename(dataset_path))

    update_plot(ax1, ax2, ax3, ax4, kms, prices, losses, r2_history, theta0, theta1, 0, nb_epochs, history, cost_grid)

    for epoch in range(1, nb_epochs + 1):

        # Update the parameters θ₀ and θ₁ using gradient descent
        tmp_theta0, tmp_theta1 = update_thetas(dataset, learning_rate, theta0, theta1)

        # θ₀ := θ₀ - η * ∂J/∂θ₀
        theta0 -= tmp_theta0
        # θ₁ := θ₁ - η * ∂J/∂θ₁
        theta1 -= tmp_theta1


        # Save the history of θ₀ and θ₁ for analysis or plotting after training
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
    signal.signal(signal.SIGINT, handler)
    dataset = read_dataset(args.dataset_path)
    theta0, theta1 = train_model(dataset, args.dataset_path)
    print(f"θ₀: {theta0}, θ₁: {theta1}")
    save_thetas(theta0, theta1)

