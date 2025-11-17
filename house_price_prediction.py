
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

np.random.seed(0)
n = 800
df = pd.DataFrame({
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 5, n),
    "area_sqft": np.random.normal(1500, 600, n).clip(300, 6000),
    "location": np.random.choice(["DHA", "Gulshan", "Clifton", "Korangi", "North Nazimabad"], n),
    "age_years": np.random.exponential(8, n),
})

base = 50_00_000
price = (base
         + df["bedrooms"]*8_00_000
         + df["bathrooms"]*5_00_000
         + df["area_sqft"]*4_000
         + df["age_years"]*(-1_50_000)
         + df["location"].map({"DHA":5_000_000,"Clifton":3_000_000,"Gulshan":1_000_000,"North Nazimabad":5_00_000,"Korangi":-3_00_000}).values
        )
df["price_pkr"] = (price + np.random.normal(0, 8_00_000, n)).clip(20_00_000, 20_000_000)

X = df.drop(columns=["price_pkr"])
y = df["price_pkr"]

num_cols = ["bedrooms","bathrooms","area_sqft","age_years"]
cat_cols = ["location"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Pipeline([
    ("pre", pre),
    ("rf", RandomForestRegressor(n_estimators=400, random_state=0))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(Xtr, ytr)
pred = model.predict(Xte)

print("MAE (PKR):", int(mean_absolute_error(yte, pred)))
print("R^2:", round(r2_score(yte, pred), 3))

rf = model.named_steps["rf"]
ohe = model.named_steps["pre"].named_transformers_["cat"]
feat_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
print("\nTop drivers:\n", importances.head(8))
