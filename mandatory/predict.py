#!/usr/bin/env python3
# Assisted-by: CLAUDE_SONNET:4.6_HIGH

# Usage : ./predict.py <km_value>

import argparse
import sys

def read_theta(filename):
    with open(filename, 'r') as f:
        return float(f.read().strip())
    
def estimate_price(km, theta0, theta1):
    return theta0 + theta1 * km

# Main
if __name__ == "__main__":
    # Reading arguments
    parser = argparse.ArgumentParser(description="Predict the price of a car based on its km value.")
    parser.add_argument(
        "km_value",
        type=int,
        help="The km value of the car for which to predict the price.",
    )
    
    # If no arguments are provided or more than 1, print help and exit
    if len(sys.argv) == 1 or len(sys.argv) > 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    km_value = args.km_value

    print(f"Predicting price for a car with {km_value} km...")
    
    # Read theta0 and theta1 from files
    theta0 = read_theta("theta0")
    theta1 = read_theta("theta1")

    # Estimate price
    predicted_price = estimate_price(km_value, theta0, theta1)
    print(f"Predicted price for a car with {km_value} km: {predicted_price}")