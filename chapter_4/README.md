# Chapter 4 — Training Models

Implementation notebooks for Chapter 4 of *Hands-On Machine Learning* (Geron), covering the mechanics of linear models: how they are trained analytically and iteratively, how they generalize, how regularization controls overfitting, and how they extend from regression to binary and multiclass classification.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [1. Linear Regression](#1-linear-regression)
- [2. Polynomial & Regularized Regression](#2-polynomial--regularized-regression)
- [3. Logistic Regression](#3-logistic-regression)
- [4. Softmax Regression](#4-softmax-regression)
- [Key Concepts Reference](#key-concepts-reference)
- [Results Summary](#results-summary)
- [Notes](#notes)

## Overview

Each notebook is self-contained, generates or loads its own dataset, and isolates one conceptual block from Chapter 4:

| Notebook | Core Idea | Model Type | Dataset |
|----------|-----------|------------|---------|
| `linear_regression.ipynb` | Closed-form vs iterative optimization | Regression | Synthetic (`y = 4 + 5x + noise`) |
| `polynomial_regression.ipynb` | Bias/variance tradeoff, regularization | Regression | Synthetic (`y = 0.5x² + x + 2 + noise`) |
| `logistic_regression.ipynb` | Binary classification via sigmoid | Classification | Iris (Virginica vs rest) |
| `softmax_regression.ipynb` | Multiclass classification via softmax | Classification | Iris (3-class) |

The progression is deliberate: start with the simplest linear model solved two different ways, add nonlinearity and the overfitting problem it introduces, then pivot from regression to classification using the same underlying linear machinery (logit -> sigmoid -> softmax).

## Repository Structure

```
chapter_4/
├── linear_regression.ipynb
├── polynomial_regression.ipynb
├── logistic_regression.ipynb
├── softmax_regression.ipynb
├── requirements.txt
└── README.md
```

No external data files are required. Synthetic data is generated with a fixed seed (`np.random.seed(42)`), and the Iris dataset is loaded directly from `sklearn.datasets`.

## Environment Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

Each notebook can be run independently, top to bottom, with no dependency on the others.

## 1. Linear Regression

**Goal:** solve the same linear regression problem four different ways and confirm they converge to the same parameters.

**Dataset:** 100 points generated from `y = 4 + 5x + Gaussian noise`, so the ground-truth parameters are known in advance (intercept ≈ 4, slope ≈ 5).

**Method 1 — Normal Equation**
Closed-form solution computed directly with NumPy:

```
theta_hat = (X^T X)^-1 X^T y
```

The bias column (a column of ones) is prepended to `X` manually (`X_b = np.c_[ones, X]`) before applying the formula. This is the only method in the notebook that does not iterate — it solves for the optimal parameters in a single matrix computation.

**Method 2 — Sklearn `LinearRegression`**
Used as a sanity check. Internally uses a different (SVD-based) computation than the raw Normal Equation but should recover near-identical intercept and coefficient values.

**Method 3 — Batch Gradient Descent**
Manual implementation. On every iteration, the gradient is computed using the *entire* dataset:

```
theta <- theta - eta * (2/m) * X^T (X*theta - y)
```

Run for 1000 iterations at a fixed learning rate (`eta = 0.1`). Because the cost surface for linear regression is convex, this reliably converges to the same optimum as the Normal Equation.

**Method 4 — Stochastic Gradient Descent**
Manual implementation that updates `theta` using a single randomly sampled instance per step rather than the full dataset, run for 50 epochs. A learning schedule (`t0 / (t + t1)`) decays the learning rate over time so the parameters settle rather than oscillate near the optimum. This is compared against sklearn's `SGDRegressor` (`max_iter=50`, no penalty) trained on the same data.

**What this notebook demonstrates:** the Normal Equation gives an exact answer instantly but scales poorly with feature count (`O(n^3)` due to the matrix inversion). Gradient descent variants scale better with `n` but require tuning (learning rate, iteration count) and only converge approximately.

## 2. Polynomial & Regularized Regression

**Goal:** show how model complexity interacts with overfitting, and how regularization and early stopping control it.

**Dataset:** 180 points generated from a quadratic relationship `y = 0.5x^2 + x + 2 + noise`, which a plain linear model cannot fit well.

**Polynomial Features**
`PolynomialFeatures(degree=2, include_bias=False)` transforms each scalar `x` into `[x, x^2]`, after which ordinary `LinearRegression` is fit on the expanded feature set. This lets a linear model capture a nonlinear relationship by making the input features nonlinear instead of the model itself.

**Learning Curves**
A helper function `plot_learning_curves(model, X, y)` trains a model on growing subsets of the training data (from 1 sample up to the full training set) and plots train vs validation RMSE at each size. Two cases are compared:

- **Degree-1 (plain linear) model:** train and validation error converge to a similarly high value — a classic underfitting signature, since a straight line cannot represent quadratic data.
- **Degree-10 polynomial model:** train error stays low while validation error remains higher and noisier — a classic overfitting signature, since the model has enough capacity to fit noise in the training data.

**Regularization**
Three penalized regression methods are fit and compared by their prediction at `x = 1.5`:

| Method | Penalty | Effect |
|--------|---------|--------|
| Ridge (`Ridge(alpha=1)`) | L2 — sum of squared weights | Shrinks all weights smoothly toward zero, none eliminated |
| Lasso (`Lasso(alpha=0.1)`) | L1 — sum of absolute weights | Drives some weights exactly to zero, performs feature selection |
| Elastic Net (`ElasticNet(alpha=0.1, l1_ratio=0.5)`) | L1 + L2 blend | Balances Ridge's stability with Lasso's sparsity |

An `SGDRegressor(penalty='l2')` is also fit as an iterative alternative to the closed-form `Ridge`.

**Early Stopping**
A degree-90 polynomial feature set is generated and standardized (`StandardScaler`), which is deliberately extreme capacity for a dataset this small. An `SGDRegressor` is trained incrementally one epoch at a time (`max_iter=1, warm_start=True`), and after each epoch the validation error is checked. The model state at the epoch with the lowest validation error is kept (`clone(sgd_reg)`) and reported as the best model — stopping training before the model starts overfitting to the training set, without needing any explicit L1/L2 penalty.

**What this notebook demonstrates:** model capacity (polynomial degree) must match the true complexity of the data. Too little capacity underfits; too much overfits unless constrained by regularization or halted early via validation monitoring.

## 3. Logistic Regression

**Goal:** binary classification — predict whether an Iris flower is *Iris Virginica* using a single feature (petal width).

**Dataset:** the standard Iris dataset from sklearn, reduced to petal width only, with the target binarized to `1` for Virginica and `0` for the other two species.

**Model**
`LogisticRegression` fits a linear decision function passed through the sigmoid:

```
p_hat = sigmoid(theta_0 + theta_1 * x) = 1 / (1 + e^-(theta_0 + theta_1 * x))
```

The fitted intercept and coefficient are printed directly after training.

**Probability Curve & Decision Boundary**
Predicted probability is computed across 1000 evenly spaced petal-width values from 0 to 3 cm, then plotted as two curves: probability of Virginica and probability of not-Virginica. The decision boundary — the petal width at which predicted probability crosses 0.5 — is located programmatically (first index where `p(not-Virginica) < 0.5`) and marked with a vertical line.

**Predictions**
The trained model is queried on two sample petal widths (1.7 cm and 1.5 cm) to show both the hard class prediction and the underlying class probabilities, illustrating how the same probability curve translates into a decision.

**What this notebook demonstrates:** logistic regression is linear regression's logit passed through a sigmoid to produce a bounded, interpretable probability, with the classification decision boundary occurring wherever that probability crosses 0.5.

## 4. Softmax Regression

**Goal:** extend the binary case to the full 3-class Iris problem using two features (petal length and petal width).

**Dataset:** the full Iris dataset (150 samples, 3 species), using only the petal length and petal width columns.

**Model**
`LogisticRegression(solver='lbfgs')` operates in multinomial mode when given more than two classes, meaning it internally learns one set of weights per class and produces probabilities via the softmax function:

```
p_hat_k = exp(s_k(x)) / sum_j exp(s_j(x))
```

where `s_k(x)` is the linear score for class `k`. The `lbfgs` solver is used since it natively supports multinomial loss, and `C=10` sets a relatively weak regularization strength.

**Prediction & Probabilities**
A sample point `[5, 2]` (petal length 5 cm, petal width 2 cm) is classified, and per-class probabilities are printed for all three species by name using `iris.target_names`.

**Decision Boundary Visualization**
A dense 2D meshgrid spanning the petal length/width feature space is classified point-by-point, then rendered as a filled contour plot (`plt.contourf`) showing the three decision regions, with the actual training points overlaid by species using distinct markers.

**What this notebook demonstrates:** softmax regression generalizes logistic regression from a single sigmoid decision boundary to `K` linear boundaries that jointly partition the feature space into `K` regions, one per class.

## Key Concepts Reference

| Concept | Where it appears | Summary |
|---------|-------------------|---------|
| Normal Equation | Linear Regression | Closed-form OLS solution, exact but `O(n^3)` in feature count |
| Batch Gradient Descent | Linear Regression | Full-dataset gradient steps, converges smoothly on convex loss |
| Stochastic Gradient Descent | Linear Regression, Early Stopping | Single-instance updates, noisier but scales to large datasets |
| Learning Schedule | Linear Regression (manual SGD) | Decaying learning rate (`t0 / (t + t1)`) to stabilize convergence |
| Polynomial Features | Polynomial Regression | Expands input features nonlinearly so a linear model can fit curves |
| Learning Curves | Polynomial Regression | Train/val error vs training set size, diagnoses under/overfitting |
| L2 Regularization (Ridge) | Polynomial Regression | Penalizes squared weight magnitude, shrinks without zeroing |
| L1 Regularization (Lasso) | Polynomial Regression | Penalizes absolute weight magnitude, induces sparsity |
| Elastic Net | Polynomial Regression | Weighted blend of L1 and L2 penalties |
| Early Stopping | Polynomial Regression | Halts training at minimum validation error, an implicit regularizer |
| Sigmoid Function | Logistic Regression | Maps linear score to a `(0, 1)` probability for binary classification |
| Decision Boundary | Logistic Regression, Softmax Regression | Input region where predicted probability crosses the classification threshold |
| Softmax Function | Softmax Regression | Generalizes sigmoid to `K` classes via normalized exponentials |

## Results Summary

- **Linear Regression:** all four methods (Normal Equation, sklearn, Batch GD, SGD) converge to parameters close to the true generating values (intercept ≈ 4, slope ≈ 5), confirming the equivalence of analytical and iterative solutions on a convex problem.
- **Polynomial Regression:** the degree-1 model underfits (high, converged train/val error), the degree-10 model overfits (low train error, higher val error), and the degree-90 model with early stopping recovers a good validation RMSE without any explicit penalty term.
- **Logistic Regression:** the fitted decision boundary falls close to a petal width of 1.6 cm, consistent with the known separability of Iris Virginica on this feature.
- **Softmax Regression:** the model cleanly partitions the 2D petal length/width space into three contiguous decision regions matching the three Iris species.

## Notes

- Synthetic datasets are seeded (`np.random.seed(42)`) for reproducibility across runs.
- The Iris dataset is loaded directly from `sklearn.datasets`, no external files required.
- Each notebook suppresses non-critical sklearn warnings (`warnings.filterwarnings('ignore')`) for cleaner output.
- Notebooks are independent — none of them import from or depend on another notebook in this folder.
