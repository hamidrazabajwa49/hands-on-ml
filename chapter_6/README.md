# Chapter 6 — Decision Trees

> Part of [Hands-On ML — From Scratch](https://github.com/hamidrazabajwa49/hands-on-ml) | Folder: `chapter_6/decision_tree`

Implementation and visualization of Decision Trees for both classification and regression, using the Iris dataset.

---

## Table of Contents
1. [Overview](#overview)
2. [Core Intuition](#core-intuition)
3. [Topics Covered](#topics-covered)
4. [Dataset](#dataset)
5. [Results](#results)
6. [Requirements](#requirements)
7. [How to Run](#how-to-run)
8. [Key Takeaways](#key-takeaways)

---

## Overview

A Decision Tree makes predictions by asking a sequence of yes/no questions about feature values, splitting the data into progressively purer groups until it reaches a decision. This notebook builds classification and regression trees on Iris petal measurements and visualizes the resulting decision boundaries.

## Core Intuition

**Think "20 questions," not equations.** A decision tree doesn't compute a formula — it asks "is petal length < 2.45?" then, depending on the answer, asks another question, and so on, until it lands on a leaf and gives its answer.

**How splits are chosen:** At each step, the tree picks the question (feature + threshold) that makes the resulting two groups as "pure" as possible — meaning each group is dominated by one class. It measures purity with **Gini impurity** (default) or entropy: a pure group (all one class) has impurity 0.

**Why `max_depth` matters:** An unrestricted tree keeps splitting until every leaf is perfectly pure — which usually means memorizing the training data (overfitting). Limiting depth forces the tree to generalize by stopping early.

**Classification vs. regression trees:** Same splitting logic, different leaf output. Classification leaves store *class probabilities* (fraction of each class among training samples that landed there). Regression leaves store the *average target value* of samples that landed there.

---

## Topics Covered

### 1. Dataset Preparation
- Iris dataset, restricted to petal length & width (2 features) for easy 2D visualization.

### 2. Decision Tree Classifier
- `DecisionTreeClassifier(max_depth=2)` — a shallow, interpretable tree.
- Reports tree depth and number of leaves.

### 3. Tree Visualization (Graphviz)
- Exports the trained tree structure to `.dot` format via `export_graphviz`.
- Render to an image with: `dot -Tpng iris_tree.dot -o iris_tree.png`
- Useful for literally seeing the "question tree" the model learned.

### 4. Predictions & Class Probabilities
- Demonstrates how a single new sample is routed through the tree to a leaf.
- `predict_proba` shows the class probability distribution stored at that leaf — this is *how* the tree "votes."

### 5. Decision Boundary Visualization
- Plots the tree's decision regions over the 2D feature space (petal length × width) as filled contour regions, overlaid with actual data points.
- Visually shows that trees produce **axis-aligned, rectangular** decision boundaries — a direct consequence of splitting on one feature at a time.

### 6. Decision Tree Regression
- `DecisionTreeRegressor(max_depth=2)` on the same features.
- Instead of predicting a class, each leaf predicts the *mean* of the target values that reached it — producing a step-function-like prediction surface.

---

## Dataset
**Iris** (`sklearn.datasets.load_iris`) — using only petal length and petal width (columns 2 and 3), across all 3 species (setosa, versicolor, virginica).

## Results

| Model | Config | Output |
|---|---|---|
| DecisionTreeClassifier | max_depth=2 | Depth and leaf count printed |
| Prediction demo | sample=[5, 1.5] | Predicted class + probability distribution |
| DecisionTreeRegressor | max_depth=2 | First 5 predictions vs. actual targets |

## Requirements

```
numpy
matplotlib
scikit-learn
graphviz (optional, for rendering .dot → .png)
```

Install via:
```bash
pip install -r requirements.txt
```

## How to Run

```bash
jupyter notebook decision_tree.ipynb
```
For tree visualization, install Graphviz separately (system package, not just the pip library) if you want to render the `.dot` file to an image.

## Key Takeaways
- Decision trees split on one feature at a time using impurity (Gini/entropy) to find the best threshold.
- `max_depth` is the primary lever against overfitting — shallow trees are more interpretable and generalize better.
- Decision boundaries are always axis-aligned rectangles — a structural limitation trees have (this motivates ensembles like Random Forests, see Chapter 7).
- Classification trees output probabilities; regression trees output averages — same mechanism, different leaf statistic.

---
*Reference: Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — Chapter 6*
