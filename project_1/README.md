# California Housing Price Prediction

End-to-end regression pipeline predicting the median house value of California census districts, using the classic 1990 California Housing dataset. 

## Problem

Predict `median_house_value` for a California census block group ("district") from 9 predictor columns: geographic coordinates, housing age, room/bedroom/population/household totals, median income, and a categorical ocean-proximity label.

## Dataset

- **Source:** California Housing dataset (1990 U.S. Census), `data/housing.csv`
- **Rows:** 20,640 districts
- **Columns:** 10 (9 features + target)
- **Missing data:** `total_bedrooms` has 207 missing values (~1%), handled via median imputation fit on the training set only
- **Target:** `median_house_value`, capped at $500,001 — see Honest Evaluation Notes

| Column | Description |
|---|---|
| `longitude`, `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms`, `total_bedrooms` | District-level totals |
| `population`, `households` | District-level totals |
| `median_income` | Median income (tens of thousands of USD) |
| `ocean_proximity` | Categorical: `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND` |
| `median_house_value` | Target variable (USD) |

## Methodology

### 1. Stratified train/test split

A plain random split can, by chance, misrepresent the income distribution in the test set, and `median_income` is the single strongest predictor of price. `median_income` is bucketed into 5 strata and `StratifiedShuffleSplit` (80/20) is used so the test set's income distribution matches the full dataset to within a fraction of a percent.

### 2. Feature engineering

Three ratio features are derived from raw district totals, generated inside the pipeline via a `FunctionTransformer` (never computed ahead of the split, so the exact same logic runs on train, CV folds, and test):

- `rooms_per_household` = `total_rooms / households`
- `population_per_household` = `population / households`
- `bedrooms_per_room` = `total_bedrooms / total_rooms`

These ratios correlate with the target more strongly than the raw totals they're derived from (e.g. `bedrooms_per_room` correlation: -0.26 vs. `total_bedrooms` alone: 0.05).

### 3. Preprocessing pipeline

Built with `ColumnTransformer`:

- **Numeric branch:** median imputation → engineered ratio features → `StandardScaler`
- **Categorical branch (`ocean_proximity`):** `OneHotEncoder(handle_unknown="ignore")`

All fitted parameters (imputation median, scaler mean/std, one-hot categories) are learned exclusively from the training set and applied unchanged to the test set — no leakage.

### 4. Model comparison (10-fold cross-validation, RMSE)

| Model | CV RMSE (mean ± std) |
|---|---|
| Linear Regression | $69,104 ± $2,880 |
| Decision Tree | $71,630 ± $2,914 |
| Random Forest | $50,436 ± $2,203 |

The Decision Tree scores *worse* than Linear Regression under cross-validation despite near-zero training error — a textbook overfitting signature, and the reason model selection here is driven by cross-validated error rather than training error.

### 5. Hyperparameter tuning

`RandomizedSearchCV` (15 iterations, 3-fold CV) over Random Forest hyperparameters (`n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`). Randomized search was chosen over an exhaustive grid to cover a wider hyperparameter space within a CPU-only compute budget.

**Best parameters:** `max_depth=23, max_features=8, min_samples_leaf=3, min_samples_split=4, n_estimators=137`
**Best CV RMSE:** $50,059

### 6. Feature importance (top 5)

| Feature | Importance |
|---|---|
| `median_income` | 0.393 |
| `ocean_proximity_INLAND` | 0.167 |
| `population_per_household` | 0.111 |
| `longitude` | 0.068 |
| `latitude` | 0.061 |

### 7. Final test set evaluation

Test set transformed with the already-fitted pipeline (no re-fitting) and scored once:

| Metric | Value |
|---|---|
| RMSE | $46,867 |
| MAE | $31,266 |
| R² | 0.8315 |

Test RMSE is lower than cross-validation RMSE, and there's no train/test performance gap indicating overfitting to the training folds.

## Repository Structure

```
california-housing-price-prediction/
├── data/
│   └── housing.csv
├── notebook.ipynb
├── README.md
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```


## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, SciPy. Runs entirely on CPU — no GPU required.
