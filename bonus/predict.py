#!/usr/bin/env python3
# Assisted-by: CLAUDE_SONNET:4.6_HIGH

# Usage : ./predict.py <km_value>

import argparse
import sys
import os

def read_theta(filename):
    if not os.path.isfile(filename) or not os.access(filename, os.R_OK):
        return None
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

    print("\nWelcome to the amazing 42's Car Price Predictor!\n")
    args = parser.parse_args()
    km_value = args.km_value
    if km_value < 0:
        print("You can only use a positive km value, please try again with a valid km value.")
        sys.exit(1)
    
    # Read theta0 and theta1 from files
    theta0 = read_theta("theta0")
    theta1 = read_theta("theta1")

    if theta0 is None or theta1 is None:
        print("Error: theta0 and theta1 must be available in the current directory.")
        sys.exit(1)

    # Estimate price
    predicted_price = estimate_price(km_value, theta0, theta1)
    if predicted_price < 0:
        print("Warning: Predicted price is negative, the km value may be out of the range of the training data.")
        sys.exit(1)
        
    # When talking about price and kilometers for cars, no need decimals
    print(f"Predicted price for a car with {int(km_value)} km: {int(predicted_price)}")
    print("\nThank you for using 42's Car Price Predictor! We hope to see you again soon!\n")