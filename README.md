<div align="center">

# 🤖 Hands-On ML — From Scratch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

Chapter-by-chapter implementations from *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Géron) — built from scratch alongside the standard library tools, with a focus on intuition and evaluation, not just calling `.fit()`.

</div>

## About

Each chapter lives in its own folder with a self-contained set of Jupyter notebooks, its own `requirements.txt`, and a detailed README covering the core intuition, topics, datasets, and results. This root README is the map — start here, then drill into whichever chapter you're interested in.

## 📂 Repository Structure

```
hands-on-ml/
├── chapter_3/          # Classification (MNIST)
├── chapter_4/          # Training Models
├── chapter_5/           
│   └── svm/             # Support Vector Machines
├── chapter_6/
│   └── decision_tree/   # Decision Trees
├── chapter_7/           # Ensemble Learning & Random Forests
│   ├── ensemble_learning/
│   └── random_forest/
├── requirements.txt      # Combined dependencies for the whole repo
└── README.md             # You are here
```

## 📖 Chapters

| # | Chapter | Focus | Key Models | README |
|---|---------|-------|------------|--------|
| 3 | Classification | Binary, multiclass, and multilabel classification on MNIST; evaluation metrics beyond accuracy | SGD, Random Forest, KNN | [chapter_3](chapter_3/) |
| 4 | Training Models | Linear/polynomial regression, regularization, logistic & softmax regression | Linear/Logistic/Softmax Regression, Ridge/Lasso/ElasticNet | [chapter_4](chapter_4/) |
| 5 | Support Vector Machines | Linear & non-linear classification, kernel trick, SVM regression | LinearSVC, SVC (poly/RBF), SVR | [chapter_5/svm](chapter_5/svm/) |
| 6 | Decision Trees | Classification & regression trees, splitting criteria, visualization | DecisionTreeClassifier/Regressor | [chapter_6/decision_tree](chapter_6/decision_tree/) |
| 7 | Ensemble Learning & Random Forests | Voting, bagging/pasting, OOB evaluation, Random Forests, AdaBoost, Gradient Boosting | Voting/Bagging/RandomForest/AdaBoost/GBRT | [chapter_7](chapter_7/) |

## 🧠 The Throughline

The chapters build on each other deliberately:

- **Ch. 3–4** establish the fundamentals: how a model is evaluated properly (accuracy is often misleading), and how linear models are trained — both analytically (Normal Equation) and iteratively (gradient descent) — before extending to classification via sigmoid/softmax.
- **Ch. 5–6** introduce two very different ways to draw a decision boundary: SVMs maximize the margin between classes (with kernels for non-linear cases), while Decision Trees split the feature space into axis-aligned rectangles.
- **Ch. 7** shows why single trees aren't the end of the story — combining many imperfect models (bagging, boosting) systematically outperforms any one of them, motivating Random Forests and Gradient Boosting as the natural evolution of Chapter 6.

## 🔧 Environment Setup

Each chapter folder has its own `requirements.txt` scoped to that chapter's dependencies. To set up everything needed across the whole repo at once, use the root-level file:

```bash
git clone https://github.com/hamidrazabajwa49/hands-on-ml.git
cd hands-on-ml
pip install -r requirements.txt
jupyter notebook
```

Then open any chapter's notebook(s) and run top to bottom. Notebooks that use `fetch_openml` (MNIST) will download and cache the dataset on first run — an internet connection is required for that step only.

## 📌 Notes

- All notebooks use `random_state=42` / `np.random.seed(42)` where randomness is involved, so results are reproducible.
- Notebooks are self-contained within each chapter — none of them import from or depend on another chapter's code.
- Tree visualization (Chapter 6) requires the Graphviz *system* package in addition to the Python library — see that chapter's README for details.

## 📚 Reference

*Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, Aurélien Géron.
