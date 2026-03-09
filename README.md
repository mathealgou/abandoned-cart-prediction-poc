# Abandoned Cart Prediction POC

This repository contains a proof of concept for predicting abandoned shopping carts using a decision tree classifier implemented in Go. The project includes data parsing, decision tree training, and prediction functionalities.

## Use cases

I created this as an exploratory exercise for demonstrating possible uses of real time customer behaviour metrics in ecommerce. The prediction (provided they are reliable enough) created by a system like this could be used for business purposes such as:

- Preventing special offers from being actively displayed at checkout for customers unlikely to stop there (Preserving margins)
- Creating and presenting coupons and offers specifically to customers who are likely to abandon their carts (Increasing conversion rates).
- Preemptively target advertisements such as whatsapp and email messages to customers with specially high likelihood of abandoning their carts, with the goal of bringing them back to the checkout page and completing their purchases (Increasing conversion rates).
- Identifying and addressing potential issues in the checkout process that may be causing customers to abandon their carts by way of measuring what points in the journey are happening when sharp increases in the chances of abandonment occur (Improving user experience).

## Features

A simple rest API is provided to make predictions based on the trained decision tree model. The API accepts JSON input with feature values and returns a prediction of whether a cart will be abandoned or not.

## Usage

1. Clone the repository:

   ```bash
   git clone
   ```

1. Navigate to the project directory:

   ```bash
   cd abandoned-cart-prediction-poc
   ```

1. Run the server (Model training is done on startup, but not persisted or chached, so it will be retrained on every server restart):

   ```bash
   go run main.go
   ```

1. Make a prediction using the API:
   ```bash
   curl -X POST http://localhost:8080/predict \
   -H "Content-Type: application/json" \
   -d '{"cart_value": 214.7, "time_on_site": 472, "pages_visited": 18}'
   ```

## Implementation Details

The decision tree is implemented from scratch in Go, without using any external machine learning libraries. The training process involves calculating the Gini impurity to find the best splits for the data and recursively building the tree until a stopping condition is met (e.g., maximum depth or pure node).

The model is trained on a sample dataset and can be easily modified to use different datasets or features by changing the data parsing and training logic.

## Note: This is a proof of concept and is not optimized for performance or accuracy. It is intended for educational purposes and will not be suitable for production use without further enhancements and very extensive testing.
