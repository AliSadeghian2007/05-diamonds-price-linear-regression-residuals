# Diamonds Price Prediction with Linear Regression

Predicting diamond prices with multiple linear regression, comparing one-hot
and ordinal encoding, and checking model fit with residual analysis.

## Dataset

[Diamonds](https://www.kaggle.com/datasets/shivam2503/diamonds) from Kaggle
(~54k rows). Download `diamonds.csv` and place it in the `data/` folder.

## Approach

1. Data inspection and cleaning (invalid rows with zero dimensions removed)
2. Exploratory data analysis (price distribution, correlations)
3. Train/test split (80/20, `random_state=42`) before any encoding
4. Model A: one-hot encoding + `LinearRegression`
5. Model B: ordinal encoding (cut, color, clarity) + `LinearRegression`
6. Model comparison and residual analysis

## Results

TODO: fill in after finishing the notebook.

| Model | R² | MAE | RMSE |
|-------|----|-----|------|
| A: One-hot | TODO | TODO | TODO |
| B: Ordinal | TODO | TODO | TODO |

**Conclusion:** TODO (which encoding worked better, and what the residual plot shows).

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```