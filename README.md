# Diamonds Price Prediction with Linear Regression

A small machine learning project that predicts diamond prices with multiple linear
regression. It compares two ways of encoding categorical features (one-hot and ordinal)
and uses residual analysis to check how well a linear model fits the data.

## Dataset

[Diamonds](https://www.kaggle.com/datasets/shivam2503/diamonds) from Kaggle. The file has
53,940 rows with these features: `carat`, `cut`, `color`, `clarity`, `depth`, `table`,
`x`, `y`, `z`. The target is `price` (in USD).

Download `diamonds.csv` and put it in the `data/` folder. The data is not stored in this
repository.

## Approach

1. **Data cleaning:** removed the index column, 146 duplicate rows and 22 rows with
   invalid dimensions (zero values or extreme outliers in `x`, `y`, `z`).
   The cleaned data has 53,772 rows.
2. **Exploratory data analysis:** price distribution, carat vs price, and a correlation
   heatmap.
3. **Train/test split:** 80/20 (`random_state=42`), done before any encoding to avoid
   data leakage. Both models use the same split.
4. **Model A:** one-hot encoding (`drop='first'`) of `cut`, `color` and `clarity`
   + `LinearRegression` (23 features).
5. **Model B:** ordinal encoding with the correct quality order + `LinearRegression`
   (9 features).
6. **Evaluation:** R², MAE and RMSE on the test set, plus residual analysis of the better
   model.

## Results

| Metric | Model A (one-hot) | Model B (ordinal) |
|--------|-------------------|-------------------|
| R² (train) | 0.921 | 0.909 |
| R² (test) | 0.919 | 0.905 |
| MAE (test) | $726.58 | $797.91 |
| RMSE (test) | $1,121.85 | $1,215.80 |

## Key Findings

- **One-hot encoding beats ordinal encoding** on every metric. Ordinal encoding assumes
  that the price difference between two neighboring grades is always the same, which is
  not true for this data. Neither model overfits.
- **`carat`, `x`, `y` and `z` are almost perfectly correlated** (0.98 to 1.0) because they
  all describe the size of the stone. This is multicollinearity, so the individual
  coefficients of these features are not reliable.
- **A straight line misses the curve in the data.** The residual plots show a curved
  pattern and larger errors for expensive stones. The model underestimates stones above
  about $12,000 and gives a few very large overestimates.
- **Model A predicts 746 negative prices** (about 7% of the 10,755 test stones), all for
  cheap stones. A linear regression has no lower bound for its predictions.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/             # put diamonds.csv here (not tracked by Git)
└── notebook.ipynb    # full analysis
```

## How to Run

```bash
git clone https://github.com/AliSadeghian2007/05-diamonds-price-linear-regression-residuals.git
cd 05-diamonds-price-linear-regression-residuals
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Tested with Python 3.13.

## Tools

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn.

## Limitations and Future Work

- A single train/test split was used. Cross-validation would give a more reliable estimate.
- The dataset has no ID column, so removing duplicate rows was a judgment call.
- Possible improvements: transform the price with a logarithm, use a model that can
  capture curves, or use regularization to handle multicollinearity.