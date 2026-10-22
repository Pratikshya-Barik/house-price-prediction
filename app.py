import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
defaults = joblib.load("defaults.pkl")

neighborhoods = sorted(c.replace("Neighborhood_", "") for c in columns if c.startswith("Neighborhood_"))

st.title("House Price Predictor")

quality = st.slider("Overall quality (1-10)", 1, 10, 5)
area = st.number_input("Living area (sq ft)", 500, 6000, 1500)
garage = st.slider("Garage capacity (cars)", 0, 4, 2)
neighborhood = st.selectbox("Neighborhood", neighborhoods)

if st.button("Predict price"):
    row = dict(defaults)
    row["OverallQual"] = quality
    row["GrLivArea"] = area
    row["GarageCars"] = garage
    row["TotalSF"] = area + defaults["TotalBsmtSF"]
    for c in columns:
        if c.startswith("Neighborhood_"):
            row[c] = 0
    row["Neighborhood_" + neighborhood] = 1

    X_new = pd.DataFrame([row])[columns]
    price = np.expm1(model.predict(X_new)[0])
    st.success(f"Estimated price: ${price:,.0f}")