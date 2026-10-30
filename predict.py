#!/usr/bin/env python3

import json


THETA_FILE = "theta.json"

def main():
    # Load trained parameters (default to 0 before training)
    theta0, theta1 = load_theta()

    # Get mileage from the user
    mileage = get_mileage()

    # Estimate the car price
    price = estimate_price(mileage, theta0, theta1)

    # Display the prediction
    print(f"Estimated price: {price:.2f} €")


if __name__ == "__main__":
    main()