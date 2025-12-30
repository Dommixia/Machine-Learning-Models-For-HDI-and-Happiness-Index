import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

#Loading dataset
df = pd.read_excel("Round 1 - Dataset - SOIL Hackathon 2025 V1.0.xlsx")

df.columns = df.columns.str.strip().str.lower()
df = df.drop(columns=["country_name"], errors="ignore")

print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. HANDLE MISSING VALUES
df = df.fillna(df.median(numeric_only=True))

target_col = "happiness_index"

le_target = LabelEncoder()
df[target_col] = le_target.fit_transform(df[target_col].astype(str))

print("\nHappiness Classes:")
for i, label in enumerate(le_target.classes_):
    print(f"{i} → {label}")

X = df.drop(columns=[target_col, "hdi_index"], errors="ignore")
y = df[target_col]

#ENCODING
X = pd.get_dummies(X, drop_first=True)

print(f"\nFinal Feature Matrix: {X.shape}")

#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#MODELS
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
}

#EVALUATION
results = []

print("\nMODEL COMPARISON")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    cv_f1 = cross_val_score(
        model, X, y, cv=5, scoring="f1_weighted"
    ).mean()

    results.append([name, acc, f1, cv_f1])

    print(f"{name:<22} | Accuracy: {acc:.4f} | F1: {f1:.4f} | CV F1: {cv_f1:.4f}")

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "F1 Score", "CV F1 Score"]
).sort_values("F1 Score", ascending=False)

print("\nFINAL RESULTS")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)

print("\nBEST MODEL:", best_model_name)
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    best_pred,
    target_names=le_target.classes_
))

if best_model_name != "Logistic Regression":
    importances = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTOP 10 FEATURES INFLUENCING HAPPINESS")
    print(importances.head(10))

    plt.figure(figsize=(10, 5))
    importances.head(10).plot(kind="bar")
    plt.title("Top Factors Influencing Happiness Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

comparison = pd.DataFrame({
    "Actual": le_target.inverse_transform(y_test),
    "Predicted": le_target.inverse_transform(best_pred)
})

print("\nPREDICTION SAMPLE")
print(comparison.head())
