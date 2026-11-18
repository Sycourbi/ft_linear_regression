# ft_linear_regression

**An introduction to machine learning — École 42**

A simple linear regression model built from scratch in Python. It learns the relationship between a car's mileage and its price using **gradient descent**, saves the trained parameters, and estimates the price of a car from its mileage.

The project also includes data visualization, a regression line, a mean-price reference, prediction-error indicators, and an **R² score**.

## Preview

![Linear regression: real prices, fitted line, mean price and prediction errors](assets/regression_graph.png)

*The blue points are the actual prices, the red line is the fitted model, the green dashed line is the average price, and the gray dotted segments show prediction errors (residuals).*

## Features

- **Train:** read a CSV file, validate its values, normalize mileage, and train a linear model using manually implemented gradient descent.
- **Predict:** ask the user for a mileage and estimate a car's price using the saved coefficients.
- **Visualize:** plot the dataset and the fitted regression line on the same graph.
- **Analyze:** display the average price, individual residuals, and the coefficient of determination (**R²**).
- **Handle errors:** check input values, CSV structure, and saved model parameters.

The regression and gradient descent are implemented **without machine-learning libraries**. Matplotlib is used only for visualization.

## Requirements

- Python 3
- `make`
- [`uv`](https://docs.astral.sh/uv/) (installed automatically by the Makefile if missing; this requires `curl` and an internet connection)
- Matplotlib (installed through `requirements.txt`)

## Installation

Clone the repository and run:

```bash
make
```

This creates a local `.venv` environment and installs the required Python dependencies using `uv`. You can also run `make install` explicitly.

## Usage

### 1. Train the model

```bash
make train data.csv
```

The program reads the dataset, runs gradient descent, and saves the resulting `theta0` and `theta1` in **`theta.json`**.

Expected output:

```text
Training completed successfully.
Model trained and saved to theta.json.
```

### 2. Predict a car's price

```bash
make predict
```

Enter the mileage when prompted:

```text
Enter car mileage: 100000
Estimated price: 6354.70 €
```

If the model has not yet been trained, both coefficients default to zero, as required by the subject. The prediction program also clamps negative estimated prices to zero.

### 3. Display the bonuses

After training, run:

```bash
make bonus data.csv
```

Expected output with the provided dataset:

```text
Model R2 score: 0.7330
```

A Matplotlib window opens with the original data, fitted regression line, average-price line, and residuals. The bonus command requires an existing `theta.json`.

## Makefile commands

| Command | Description |
| --- | --- |
| `make` | Set up the project and install dependencies |
| `make install` | Create `.venv` and install requirements with `uv` |
| `make train data.csv` | Train the model on a CSV dataset |
| `make predict` | Predict a price from a mileage entered by the user |
| `make bonus data.csv` | Show the graph and print the R² score |
| `make clean` | Remove the saved model (`theta.json`) |
| `make fclean` | Remove the saved model and `.venv` |
| `make re` | Clean and reinstall the project |

## How it works

### 1. Linear regression

The model predicts a price using a single feature, the car's mileage:

\[
\widehat{price} = \theta_0 + \theta_1 \times mileage
\]

- **θ₀** is the intercept: the predicted price at zero mileage.
- **θ₁** is the slope: how much the estimated price changes for each additional kilometer.

### 2. Data normalization

Before training, mileage is standardized to make gradient descent more stable:

\[
x_{normalized} = \frac{x - \mu}{\sigma}
\]

Here, **μ** is the mean mileage and **σ** is its standard deviation. After training, the parameters are converted back to work with actual kilometers. This normalization does not require NumPy or scikit-learn.

### 3. Gradient descent

The model starts with `theta0 = 0` and `theta1 = 0` and repeatedly adjusts them to reduce the difference between predicted and actual prices.

For each training iteration:

\[
\theta_0 \leftarrow \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^{m}(\widehat{y_i}-y_i)
\]

\[
\theta_1 \leftarrow \theta_1 - \alpha\frac{1}{m}\sum_{i=1}^{m}(\widehat{y_i}-y_i)x_i
\]

In these formulas, **m** is the number of cars, **α** is the learning rate, and **xᵢ** is the **normalized mileage during training**. Both coefficients are updated simultaneously using the same iteration's gradients.

Current training settings:

| Parameter | Value |
| --- | ---: |
| Learning rate | `0.01` |
| Iterations | `10000` |
| Training feature | Mileage (`km`) |
| Target | Price (`price`) |

### 4. Saved model

After training, the coefficients are stored in `theta.json` and reused by `predict.py` and `visualize.py`. For the supplied dataset, they are approximately:

```json
{
  "theta0": 8499.599649933149,
  "theta1": -0.021448963591702085
}
```

These values depend on the dataset used for training.

## Model evaluation: R²

The coefficient of determination compares the model's squared prediction errors with the squared deviations from the mean price:

\[
R^2 = 1 - \frac{\sum_i(y_i-\widehat{y_i})^2}{\sum_i(y_i-\bar{y})^2}
\]

On the supplied **24-car dataset**:

| Metric | Result |
| --- | ---: |
| Number of cars | `24` |
| Average car price | `6331.83 €` |
| R² score | **`0.7330`** |

An R² of approximately **0.7330** means that the fitted line explains about **73.3% of the variation in the prices in this dataset**. It does **not** mean that 73.3% of individual predictions are correct, and it is **not** a measure of performance on unseen data.

### Reading the graph

- **Blue dots — Real prices:** observations from `data.csv`.
- **Red line — Regression line:** predictions computed from the trained coefficients.
- **Green dashed line — Mean price:** the average observed price, used as a baseline for understanding R².
- **Gray dotted segments — Residuals:** vertical differences between observed and predicted prices; longer segments indicate larger errors.

## Dataset format

The program expects a CSV file with exactly these column names, in this order:

```csv
km,price
240000,3650
139800,3800
150500,4400
```

The loader checks that the file has a `.csv` extension, contains at least two observations, has nonnegative finite numeric values, and includes different mileage values.

You can train on another dataset with the same structure:

```bash
make train other_data.csv
make bonus other_data.csv
```

**Note:** `make bonus` evaluates the currently saved model against the CSV you provide. For meaningful results, use the same dataset you used to train it unless you intentionally want to evaluate on another dataset.

## Project structure

```text
ft_linear_regression/
├── Makefile
├── README.md
├── requirements.txt
├── data.csv
├── train.py
├── predict.py
├── visualize.py
├── assets/
│   └── regression_graph.png
├── theta.json          # Generated after training
└── .venv/              # Generated by make
```

| File | Role |
| --- | --- |
| `train.py` | CSV validation, normalization, gradient descent, model saving |
| `predict.py` | Model loading, input validation, price prediction |
| `visualize.py` | Scatter plot, regression line, mean price, residuals, R² |
| `Makefile` | Installation and execution shortcuts |
| `requirements.txt` | Matplotlib dependency |
| `data.csv` | Example training dataset |

## Subject requirements

**Mandatory:**

- [x] Predict the price of a car from its mileage.
- [x] Train a linear regression model using gradient descent.
- [x] Initialize the parameters at zero before training.
- [x] Update both parameters simultaneously.
- [x] Save and reuse the trained coefficients.

**Bonuses:**

- [x] Plot the dataset distribution.
- [x] Draw the regression line on the same graph.
- [x] Calculate a model-precision metric (R²).

Additional visualization: mean-price reference line and prediction residuals.

## Limitations

- The model uses **mileage only**; car prices also depend on age, condition, brand, and other factors.
- R² is computed on the supplied dataset, not on a separate test set.
- The prediction CLI clips negative estimated prices to zero, while the graph and R² use the raw linear formula.
- Extrapolating far beyond the mileage range in the dataset may produce unreliable estimates.

## Acknowledgments

Created as part of the **École 42 `ft_linear_regression`** project (subject version 4.1). AI assistance was used to review code, discuss tests, and help prepare documentation; implementation decisions and understanding remain the student's responsibility.
