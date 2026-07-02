# Chapter 7 — Ensemble Learning & Random Forests

> Part of [Hands-On ML — From Scratch](https://github.com/hamidrazabajwa49/hands-on-ml) | Folder: `chapter_7/`

This chapter explores ensemble methods — combining multiple models to achieve better predictive performance than any single model alone. It covers the two major families: **bagging** (with a deep dive into Random Forests) and **boosting** (AdaBoost and Gradient Boosting). All implementations are built from scratch using scikit-learn and include a manual walkthrough of Gradient Boosting to build intuition.

---

## Table of Contents
- [Overview](#overview)
- [Core Intuition](#core-intuition)
- [Repository Structure](#repository-structure)
- [Topics Covered](#topics-covered)
- [Datasets](#datasets)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Key Takeaways](#key-takeaways)

---

## Overview

The "wisdom of the crowd" principle applied to ML: a group of moderately good, sufficiently *different* models can outperform any single strong model — provided their errors aren't all correlated. This chapter builds that intuition through:

- **Voting classifiers** that combine different algorithms.
- **Bagging** and **Pasting** that train the same algorithm on random subsets.
- **Random Forests** — a specialized form of bagged decision trees with extra randomness at each split.
- **Boosting** — sequential methods (AdaBoost, Gradient Boosting) that correct previous mistakes.
- **Feature importance** as a free byproduct of tree-based ensembles.
- **Manual implementation** of Gradient Boosting to see the algorithm in action.

---

## Core Intuition

**Why ensembles work:** If you have several models that are each slightly better than random guessing, and their mistakes are somewhat independent, combining their votes cancels out individual errors — much like averaging many noisy measurements gets you closer to the true value.

**The critical condition — diversity:** Ensembling only helps if the models actually disagree. Diversity is created either by using *different algorithms* (Voting Classifier) or by training the *same algorithm on different random subsets of data* (Bagging/Pasting) or by adding randomness to the training process (Random Forest).

**Bagging vs. Boosting — the fundamental split in ensemble learning:**
- **Bagging (Random Forest):** trains all models *independently and in parallel*; they don't know about each other. Reduces **variance**.
- **Boosting (AdaBoost, Gradient Boosting):** trains models *sequentially*; each new model is built specifically to fix the previous ensemble's mistakes. Reduces **bias**.

**Out-of-Bag (OOB) evaluation — the "free lunch":** Because bagging samples with replacement, about 37% of the training data is left out of each bootstrap sample. Those left‑out points act as a natural validation set — no need for a separate holdout split.

**Random Forest = Bagging + extra randomness.** A plain bagged ensemble of trees already reduces variance. Random Forest adds one more twist: each tree, at each split, only gets to consider a *random subset of features* — not all of them. This forces trees to be even more different, further reducing correlation between their errors and improving the ensemble.

**Feature importance — a free byproduct:** Because Random Forest tracks how much each feature reduces impurity across all trees and splits, it can rank features by importance automatically — no separate analysis needed. This is why it's often used for feature selection, not just prediction.

**AdaBoost intuition:** After each weak learner is trained, AdaBoost *increases the weight* of the training examples it got wrong — so the next learner is forced to pay more attention to the hard cases. The final prediction is a weighted vote, where more accurate learners get a bigger say.

**Gradient Boosting intuition — the real "aha":** Instead of reweighting examples, Gradient Boosting trains each new tree to predict the **residual errors** (actual − predicted) of the current ensemble. Add that tree's predictions to the ensemble, compute new residuals, repeat. This is literally gradient descent, but instead of adjusting parameters, you're adding a new tree at each step that points the ensemble further downhill.

---

## Repository Structure

This chapter is split into **two complementary notebooks**:

| Notebook | Focus |
|---|---|
| [`ensemble_learning/ensemble_learning.ipynb`](ensemble_learning/ensemble_learning.ipynb) | Voting, Bagging, Pasting, Out‑of‑Bag evaluation |
| [`random_forest/random_forest.ipynb`](random_forest/random_forest.ipynb) | Random Forests (feature importance, MNIST pixel importance), AdaBoost, Gradient Boosting (manual + sklearn) |

Both notebooks are self‑contained and can be run independently.

---

## Topics Covered

### 1. Ensemble Learning (Notebook 1)
- **Voting Classifier:** Hard voting vs. soft voting with Logistic Regression, Random Forest, and SVM.
- **Bagging:** `BaggingClassifier` with 500 decision trees, bootstrap sampling.
- **Pasting:** Same as bagging but without replacement (`bootstrap=False`).
- **Out‑of‑Bag (OOB) evaluation:** Use `oob_score=True` to get a built‑in validation estimate.

### 2. Random Forests (Notebook 2)
- `RandomForestClassifier` on `make_moons` with 500 trees and restricted leaf nodes.
- **Feature Importance on Iris:** Ranks the four features by their contribution to impurity reduction.
- **Pixel Importance on MNIST:** Trains on 70k images, reshapes `feature_importances_` into a 28×28 heatmap showing which pixels matter most for digit classification.

### 3. Boosting (Notebook 2)
- **AdaBoost:** Uses decision stumps (depth‑1 trees) as weak learners and combines 200 of them.
- **Gradient Boosting — Manual:** Builds a 3‑tree ensemble by hand, fitting each new tree to the residuals of the current ensemble.
- **Gradient Boosting — sklearn:** Confirms the manual implementation via `GradientBoostingRegressor`.
- **Early Stopping (staged prediction):** Trains a large ensemble and uses `staged_predict` to find the optimal number of trees.
- **Incremental Early Stopping:** More efficient: train one tree at a time (`warm_start=True`) and stop when validation error fails to improve for 5 consecutive rounds.

---

## Datasets

| Dataset | Usage | Notebook |
|---|---|---|
| `make_moons` (500 samples, noise=0.30) | Classification for Voting, Bagging, Random Forest, AdaBoost | Both |
| Iris (4 features, 3 classes) | Feature importance demo | random_forest |
| MNIST (70k images, 784 pixels) | Pixel importance heatmap | random_forest |
| Synthetic quadratic + noise | Gradient Boosting regression demos | random_forest |

---

## Results

| Method | Task | Metric / Output | Notebook |
|---|---|---|---|
| Voting Classifier (soft) | make_moons | Test accuracy (≥ best individual) | ensemble_learning |
| Bagging (500 trees) | make_moons | Test accuracy | ensemble_learning |
| Pasting (500 trees) | make_moons | Test accuracy | ensemble_learning |
| Bagging + OOB | make_moons | OOB score vs. test accuracy | ensemble_learning |
| Random Forest (500 trees) | make_moons | Test accuracy | random_forest |
| Random Forest | Iris | Ranked feature importances | random_forest |
| Random Forest | MNIST | Pixel importance heatmap (visual) | random_forest |
| AdaBoost (200 stumps) | make_moons | Test accuracy | random_forest |
| Manual GBRT (3 trees) | Synthetic regression | Predictions at test points | random_forest |
| Sklearn GBRT | Synthetic regression | Predictions (should match manual) | random_forest |
| GBRT + staged early stopping | Synthetic regression | Optimal `n_estimators` + validation MSE | random_forest |
| GBRT + incremental stopping | Synthetic regression | Early‑stop iteration + validation MSE | random_forest |

*Run the notebooks for exact numbers — all results are reproducible with `random_state=42`.*

---

## Requirements
