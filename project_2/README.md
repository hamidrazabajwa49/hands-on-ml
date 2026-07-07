<div align="center">

# 🔢 MNIST Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

Binary, multiclass, and multilabel classification on handwritten digits — with a focus on evaluation metrics, not just accuracy.

</div>

## TL;DR

| | |
|---|---|
| **Task** | Binary (digit-5 detector), multiclass (0–9), multilabel (large / odd) |
| **Dataset** | MNIST 784 (70,000 images, 28×28 px) via OpenML |
| **Best binary model** | Random Forest — AUC ≈ 0.99 (vs SGD ≈ 0.96) |
| **Multiclass models** | SGD (OvR), OvO-SGD (45 estimators), Random Forest |
| **Multilabel model** | KNN, macro F1 reported per label |
| **Key finding** | Accuracy alone is misleading on the imbalanced 5-detector (90% baseline from predicting "not 5") |

## 📂 Dataset

<details>
<summary>Click to expand</summary>

- **Source:** `sklearn.datasets.fetch_openml('mnist_784')`
- **Size:** 70,000 grayscale images, 28×28 px (784 features), flattened to pixel-intensity vectors
- **Labels:** digits 0–9
- **Split:** first 60,000 = train (shuffled to break ordering), last 10,000 = test — this is MNIST's standard, pre-defined split, not a random one
- **Derived targets used in this notebook:**
  - `y_train_5`: binary, digit == 5
  - `y_train` (multiclass): original 0–9 labels
  - `y_multilabel`: two binary columns — `is_large` (digit ≥ 7), `is_odd`

</details>

## 🔧 Pipeline

```mermaid
flowchart LR
    A[Load MNIST 70k] --> B[Train/Test Split 60k/10k]
    B --> C[Shuffle Train Set]
    C --> D[Binary: digit-5 detector]
    C --> E[Multiclass: 0-9]
    C --> F[Multilabel: large/odd]
    D --> D1[SGD + CV Accuracy]
    D1 --> D2[Confusion Matrix, Precision/Recall/F1]
    D2 --> D3[PR Curve, ROC/AUC]
    D3 --> D4[SGD vs Random Forest ROC]
    E --> E1[SGD OvR]
    E --> E2[OvO-SGD, 45 estimators]
    E --> E3[Random Forest]
    E1 --> E4[Normalized Confusion Matrix]
    F --> F1[KNN multilabel fit]
    F1 --> F2[Cross-val F1 per label + macro]
```

## 📊 Results

**Binary (digit-5 detector), 3-fold CV:**

| Metric | Value |
|---|---|
| SGD Accuracy | ~0.965 |
| Dummy baseline Accuracy | ~0.910 |
| Precision | 0.824 |
| Recall | 0.785 |
| F1 | 0.804 |
| SGD AUC | ~0.96 |
| Random Forest AUC | ~0.99 |

**Multiclass:** SGD (OvR, native), OvO-SGD (45 pairwise classifiers), Random Forest — evaluated with a normalized, diagonal-zeroed confusion matrix to highlight which digit pairs get confused (e.g. 4↔9, 3↔5 are the classic MNIST error pairs).

**Multilabel (KNN):** predicts `is_large` (≥7) and `is_odd` simultaneously; scored with per-label F1 and macro F1 via 3-fold `cross_val_predict`.


## 🚀 Usage

```bash
pip install -r requirements.txt
jupyter notebook mnist.ipynb
```

Run all cells top to bottom. `fetch_openml` downloads and caches MNIST on first run (needs internet).
