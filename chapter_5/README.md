# Chapter 5 — Support Vector Machines (SVM)

> Part of [Hands-On ML — From Scratch](https://github.com/hamidrazabajwa49/hands-on-ml) | Folder: `chapter_5/svm`

Implementation and exploration of Support Vector Machines for classification and regression, covering linear separation, kernel tricks, and margin-based regression.

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

SVM finds a decision boundary that doesn't just separate classes — it separates them by the **widest possible margin**. This notebook builds intuition for linear SVMs, extends to non-linear data via kernels, and closes with SVM regression (SVR).

## Core Intuition

**Why "maximum margin" matters:** Many lines can separate two classes. SVM doesn't pick any separating line — it picks the one that stays as far as possible from the nearest points of *both* classes. This makes the model more robust to new, unseen data landing near the boundary.

**Support vectors:** Only the points closest to the boundary ("support vectors") determine where that boundary sits. Points far from the boundary could be deleted entirely and the boundary wouldn't move — this is what makes SVM memory-efficient at decision time.

**The kernel trick (the key "aha"):** Some data isn't linearly separable in its original space (e.g., two interleaving moons). Instead of manually engineering new features to make it separable, kernels compute what the similarity *would be* in a higher-dimensional space — without ever actually transforming the data. You get the benefit of extra dimensions without the computational cost of creating them.

---

## Topics Covered

### 1. Linear SVM Classification
- **What:** `LinearSVC` with hinge loss on the Iris dataset (petal length/width), classifying *Iris-Virginica* vs. rest.
- **Why it matters:** Establishes the baseline — maximum-margin classification when data is (roughly) linearly separable.
- **Key parameter:** `C` — controls the trade-off between a wide margin and margin violations. Low `C` = wider margin, more tolerance for misclassified points (softer). High `C` = narrower margin, fewer violations (stricter, risk of overfitting).

### 2. Non-Linear SVM Classification
Dataset: `make_moons` (500 samples, noise=0.30) — two interleaving crescents, not linearly separable.

| Approach | Method | Idea |
|---|---|---|
| **2a. Polynomial Features** | `PolynomialFeatures(degree=3)` + `LinearSVC` | Manually expand features into polynomial combinations, then draw a straight line in that expanded space |
| **2b. Polynomial Kernel** | `SVC(kernel='poly')` | Achieves the same effect as above *without* explicitly creating new features (kernel trick) |
| **2c. RBF (Gaussian) Kernel** | `SVC(kernel='rbf', gamma=...)` | Measures similarity to "landmark" points; `gamma` controls how localized each point's influence is |

**Gamma intuition:** Think of `gamma` as a magnifying glass. Low gamma = each point influences a wide area (smoother, more generalized boundary). High gamma = each point only influences its immediate neighborhood (tighter, wigglier boundary — overfitting risk).

### 3. SVM Regression (SVR)
- **Flip the objective:** Classification SVM maximizes the margin *between* classes. SVR instead tries to fit as many points as possible *inside* a margin (the "epsilon-tube") around the predicted line.
- **3a. LinearSVR:** Linear regression with an epsilon-insensitive margin — errors within `epsilon` aren't penalized at all.
- **3b. Polynomial Kernel SVR:** Same margin idea, but the underlying function can be curved (via the poly kernel).

---

## Dataset
- **Iris** (`sklearn.datasets.load_iris`) — linear classification demo
- **make_moons** — non-linear classification demo (500 samples, noise=0.30)
- **Synthetic linear/noisy data** — regression demos

## Results

| Model | Task | Metric |
|---|---|---|
| Linear SVC | Iris (Virginica vs rest) | Correct prediction on test point [5.5, 1.7] |
| Polynomial Features + LinearSVC | make_moons | Accuracy printed on training data |
| Polynomial Kernel SVC | make_moons | Accuracy printed on training data |
| RBF Kernel SVC | make_moons | Accuracy printed on training data |
| LinearSVR | Synthetic regression | Prediction at x=0.5 |
| Polynomial Kernel SVR | Synthetic regression | Prediction at x=0.5 |

*Note: accuracy values are computed on training data in this notebook — run cells to see live numbers. For production evaluation, use a held-out test set.*

## Requirements

```
numpy
matplotlib
scikit-learn
```

Install via:
```bash
pip install -r requirements.txt
```

## How to Run

```bash
jupyter notebook svm.ipynb
```
Run cells top to bottom. Each section is self-contained with markdown explanations preceding the code.

## Key Takeaways
- Linear SVM = maximum margin classifier; controlled by `C`.
- Non-linear separability is handled via feature expansion (explicit) or kernels (implicit, computationally cheaper).
- RBF kernel's `gamma` and `C` jointly control the bias-variance trade-off — always tune together.
- SVR reframes the SVM objective: fit points *inside* a margin rather than separate classes *across* one.

---
*Reference: Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — Chapter 5*
