import numpy as np
import joblib
from data_prep import load_data, clean_data, build_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    df = load_data("data/train.csv")
    df = clean_data(df)
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, np.log1p(y_train))

    preds = np.expm1(model.predict(X_test))
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))

    joblib.dump(model, "model.pkl")
    joblib.dump(list(X.columns), "columns.pkl")
    joblib.dump(X.median().to_dict(), "defaults.pkl")

if __name__ == "__main__":
    main()
    