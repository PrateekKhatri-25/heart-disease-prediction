
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. Paths and dataset loading
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleveland.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
RESULTS_PATH = BASE_DIR / "model_comparison.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Save cleveland.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Remove an accidental exported index column, if present.
df = df.loc[
    :, ~df.columns.str.match(r"^Unnamed:")
].copy()

# Support either the original target name or a binary target.
if "num" in df.columns:
    target_column = "num"
elif "target" in df.columns:
    target_column = "target"
else:
    raise ValueError(
        "Expected a target column named 'num' or 'target'. "
        f"Available columns: {df.columns.tolist()}"
    )

# Validate the expected Cleveland feature columns.
categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

numeric_features = [
    "age", "trestbps", "chol",
    "thalach", "oldpeak"
]

feature_columns = numeric_features + categorical_features
missing_columns = [
    col for col in feature_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing expected features: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Convert feature values to numeric. Invalid entries become NaN.
for col in feature_columns + [target_column]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# The target must be present.
df = df.dropna(subset=[target_column]).copy()

# Convert the original target (0, 1, 2, 3, 4)
# into binary classification: 0 versus greater than 0.
df["target"] = (df[target_column] > 0).astype(int)

X = df[feature_columns].copy()
y = df["target"].copy()

if y.nunique() != 2:
    raise ValueError(
        "The target must contain both classes: 0 and 1."
    )

print("Dataset:", DATA_PATH)
print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(y.value_counts().sort_index())
print("\nMissing values:")
print(X.isna().sum())


# --------------------------------------------------
# 2. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 3. Reusable preprocessing pipeline
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# --------------------------------------------------
# 4. Define three models and tuning parameters
# --------------------------------------------------

model_configs = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=2000),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__class_weight": [None, "balanced"],
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [1, 3, 5],
            "model__class_weight": [None, "balanced"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(
            random_state=42,
            n_jobs=1,
        ),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 3],
            "model__class_weight": [None, "balanced"],
        },
    },
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# --------------------------------------------------
# 5. Tune and compare all three models
# --------------------------------------------------

results = []
best_cv_score = -1
best_search = None
best_name = None

for name, config in model_configs.items():
    print(f"\nTuning {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", config["estimator"]),
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["params"],
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )

    # Only training data is used for CV and tuning.
    search.fit(X_train, y_train)

    results.append({
        "Model": name,
        "Best CV F1": search.best_score_,
        "Best Parameters": str(search.best_params_),
    })

    print(f"Best CV F1: {search.best_score_:.4f}")
    print("Best parameters:", search.best_params_)

    if search.best_score_ > best_cv_score:
        best_cv_score = search.best_score_
        best_search = search
        best_name = name


# --------------------------------------------------
# 6. Final evaluation on the held-out test set
# --------------------------------------------------

final_pipeline = best_search.best_estimator_
predictions = final_pipeline.predict(X_test)

print("\n" + "=" * 50)
print("FINAL MODEL:", best_name)
print("=" * 50)

print(f"Best CV F1: {best_cv_score:.4f}")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
print(
    f"Test precision: "
    f"{precision_score(y_test, predictions, zero_division=0):.4f}"
)
print(
    f"Test recall: "
    f"{recall_score(y_test, predictions, zero_division=0):.4f}"
)
print(
    f"Test F1: "
    f"{f1_score(y_test, predictions, zero_division=0):.4f}"
)

if hasattr(final_pipeline, "predict_proba"):
    probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    print(
        f"Test ROC-AUC: "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Class 0", "Class 1"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))


# --------------------------------------------------
# 7. Save the fitted pipeline and comparison results
# --------------------------------------------------

joblib.dump(final_pipeline, MODEL_PATH)

results_df = pd.DataFrame(results).sort_values(
    "Best CV F1",
    ascending=False,
)

results_df.to_csv(RESULTS_PATH, index=False)

print(f"\nSaved model: {MODEL_PATH}")
print(f"Saved comparison: {RESULTS_PATH}")
print("\nModel comparison:")
print(results_df.to_string(index=False))
