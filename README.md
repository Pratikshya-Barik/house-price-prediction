# House Price Prediction
**Live demo:** https://house-price-prediction-ds2c7j4zyrseywpdjcq2sk.streamlit.app

A machine learning project that predicts house sale prices from the Ames Housing dataset (Kaggle: "House Prices - Advanced Regression Techniques").

## Results
| Model | 5-fold CV R² |
|---|---|
| Linear Regression | 0.80 |
| Random Forest | 0.86 |

On a held-out test set (20% of the data), the final Random Forest reached MAE ≈ $16,600 and R² ≈ 0.90.

With the log transform and outlier removal, 5-fold cross-validated R² rose from 0.86 to 0.88.

## How it works
1. **Exploration:** looked at the distribution of `SalePrice` (right-skewed), its relationship with features such as living area, price differences across neighborhoods, outliers, and missing values.
2. **Cleaning:** many "missing" values in this dataset mean the house doesn't have that feature (no garage, no pool, no alley), so these were filled with a `"None"` category. Genuinely unknown values were filled with the median (`LotFrontage`), 0 (`MasVnrArea`) or the most common value (`Electrical`). The `Id` and `GarageYrBlt` columns were dropped.
3. **Encoding:** text columns were converted to numbers with one-hot encoding, giving 301 input features.
4. **Models:** a baseline Linear Regression (R² 0.79 with 6 hand-picked features, 0.89 on a single split with all features) was compared with a Random Forest.
5. **Evaluation:** single train/test splits turned out to be misleading, so models were compared with 5-fold cross-validation. Linear Regression was unstable across folds (R² from 0.65 to 0.89), while the Random Forest was steadier (0.82 to 0.89).
6. **Improvements:** applied a log transform to `SalePrice` (reduces skew) and removed two outlier houses (very large but cheap, both incomplete "Partial" sales). Together these lowered the test MAE from about $17,900 to about $16,600.

## Things I tried that didn't help
- A `TotalSF` feature (basement + first floor + second floor area) did not improve Linear Regression, since the model can already combine those columns itself.
- Tuning the number of trees and the maximum depth with GridSearchCV found the default settings were already the best.

## Project structure
- `data/`: dataset
- `src/data_prep.py`: loading, cleaning and building features
- `src/train.py`: trains and evaluates the model
- `explore.ipynb`: data exploration and experiments
- `requirements.txt`: required libraries

## How to run
    pip install -r requirements.txt
    python src/train.py

## Possible improvements
- Try other models (for example gradient boosting) and tune more settings, such as `min_samples_leaf` and `max_features`
- Add more engineered features and check them with cross-validation
- Re-check the log transform and the outlier removal with cross-validation, since they were only tested on a single split
- Build a simple web app where a user enters house details and gets a predicted price
