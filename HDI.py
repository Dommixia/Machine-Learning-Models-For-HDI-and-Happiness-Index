import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, r2_score


#LOADING DATASET
df = pd.read_excel("Round 1 - Dataset - SOIL Hackathon 2025 V1.0.xlsx")

df.columns = df.columns.str.strip().str.lower()
df = df.drop(columns=["country_name"], errors="ignore")


ordinal_cols = [
    "income_group",
    "education_level",
    "healthcare_quality",
    "development_status"
]

for col in ordinal_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Fill missing values
df = df.fillna(df.median(numeric_only=True))


X = df.drop("hdi_index", axis=1)
y = df["hdi_index"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        random_state=42
    )
}

#EVALUATION
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    results.append([name, mae, r2, cv_r2])

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "Test R²", "CV R²"]
)

print("\nMODEL COMPARISON")
print("=" * 50)
print(results_df.sort_values("Test R²", ascending=False))
print("=" * 50)

best_model = models["Gradient Boosting"]

importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTOP 10 HDI INFLUENCING FEATURES")
print(importances.head(10))

y_pred_best = best_model.predict(X_test)

plt.figure()
plt.scatter(y_test, y_pred_best)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--")
plt.xlabel("Actual HDI")
plt.ylabel("Predicted HDI")
plt.title("Actual vs Predicted HDI (Best Model)")
plt.show()
