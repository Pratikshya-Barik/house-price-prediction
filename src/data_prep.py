import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.copy()
    df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))]

    # Columns where missing means "doesn't have it"
    none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "MasVnrType",
                 "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                 "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    df[none_cols] = df[none_cols].fillna("None")

    # Columns where the value is genuinely unknown
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    # Drop columns we decided not to use
    df = df.drop(columns=["Id", "GarageYrBlt"])

    # The total square footage column we built earlier
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    return df


def build_features(df):
    df_encoded = pd.get_dummies(df)
    X = df_encoded.drop(columns=["SalePrice"])
    y = df_encoded["SalePrice"]
    return X, y