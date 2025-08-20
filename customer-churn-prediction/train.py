# train.py — trains multiple models, picks best by F1, saves a single Pipeline
import argparse, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
import joblib

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Optional XGBoost
HAVE_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAVE_XGB = False
    warnings.warn("xgboost not installed — skipping XGB.")

# Optional TensorFlow ANN
HAVE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    tf.random.set_seed(42)
except Exception:
    HAVE_TF = False
    warnings.warn("TensorFlow/Keras not installed — skipping ANN.")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------- helpers --------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def detect_target(df: pd.DataFrame, target_arg: str | None) -> str:
    if target_arg:
        t = target_arg.lower()
        if t not in df.columns:
            raise ValueError(f"Target '{t}' not found. Columns: {list(df.columns)[:10]}...")
        return t
    for cand in ["churn", "exited", "target", "label", "y"]:
        if cand in df.columns:
            return cand
    raise ValueError("Could not infer target. Use --target <colname>.")

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre

def fit_and_score(pipe, X_tr, y_tr, X_te, y_te, grid=None):
    if grid is None:
        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te)
        return None, f1_score(y_te, y_hat), accuracy_score(y_te, y_hat), pipe

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    g = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True, verbose=0)
    g.fit(X_tr, y_tr)
    y_hat = g.predict(X_te)
    return g.best_params_, f1_score(y_te, y_hat), accuracy_score(y_te, y_hat), g.best_estimator_

def ann_estimator(input_dim: int):
    model = Sequential([
        Dense(16, activation="relu", kernel_initializer="glorot_uniform", input_dim=input_dim),
        Dense(8, activation="relu", kernel_initializer="glorot_uniform"),
        Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Data.csv")
    ap.add_argument("--target", default=None)
    ap.add_argument("--with-ann", action="store_true", help="try ANN if TF is installed")
    args = ap.parse_args()

    models_dir = Path("models"); models_dir.mkdir(exist_ok=True)

    df = load_data(args.data)

    # Avoid leakage from obvious IDs/PII
    EXCLUDE = ["phone_number"]
    df = df.drop(columns=[c for c in EXCLUDE if c in df.columns])

    target = detect_target(df, args.target)
    y = df[target].astype(int).to_numpy().ravel()
    X = df.drop(columns=[target])

    pre = build_preprocessor(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    artifacts = {}

    # 1) Logistic Regression
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    grid = {"clf__C": [0.1, 1, 10], "clf__solver": ["lbfgs"], "clf__penalty": ["l2"]}
    bp, f1, acc, est = fit_and_score(pipe, X_tr, y_tr, X_te, y_te, grid)
    results.append(("logreg", f1, acc, bp)); artifacts["logreg"] = est

    # 2) Random Forest
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
    grid = {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 8, 12], "clf__min_samples_split": [2, 5]}
    bp, f1, acc, est = fit_and_score(pipe, X_tr, y_tr, X_te, y_te, grid)
    results.append(("rf", f1, acc, bp)); artifacts["rf"] = est

    # 3) Decision Tree
    pipe = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    grid = {"clf__max_depth": [None, 6, 10], "clf__min_samples_split": [2, 5, 10]}
    bp, f1, acc, est = fit_and_score(pipe, X_tr, y_tr, X_te, y_te, grid)
    results.append(("dt", f1, acc, bp)); artifacts["dt"] = est

    # 4) KNN
    pipe = Pipeline([("pre", pre), ("clf", KNeighborsClassifier())])
    grid = {"clf__n_neighbors": [5, 15, 25], "clf__weights": ["uniform", "distance"]}
    bp, f1, acc, est = fit_and_score(pipe, X_tr, y_tr, X_te, y_te, grid)
    results.append(("knn", f1, acc, bp)); artifacts["knn"] = est

    # 5) SVM (set probability=True if you want calibrated probs)
    pipe = Pipeline([("pre", pre), ("clf", SVC(probability=False, random_state=RANDOM_STATE))])
    grid = {"clf__C": [0.5, 1, 5], "clf__kernel": ["rbf", "linear"], "clf__gamma": ["scale"]}
    bp, f1, acc, est = fit_and_score(pipe, X_tr, y_tr, X_te, y_te, grid)
    results.append(("svm", f1, acc, bp)); artifacts["svm"] = est

    # 6) XGBoost (if available)
    if HAVE_XGB:
        pipe = Pipeline([("pre", pre), ("clf", XGBClassifier(
            random_state=RANDOM_STATE, n_estimators=300, eval_metric="logloss", n_jobs=-1, tree_method="hist"
        ))])
        grid = {"clf__max_depth": [4, 5, 6], "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8, 1.0], "clf__colsample_bytree": [0.8, 1.0]}
        bp, f1, acc, est = fit_and_score(pipe, X_tr, y_tr, X_te, y_te, grid)
        results.append(("xgb", f1, acc, bp)); artifacts["xgb"] = est

    # 7) ANN (optional; only if you pass --with-ann AND TF installed)
    if args.with_ann and HAVE_TF:
        _pre = build_preprocessor(X_tr); _pre.fit(X_tr)
        input_dim = _pre.transform(X_tr[:5]).shape[1]
        model = ann_estimator(input_dim)
        Xtr_ = _pre.transform(X_tr); Xte_ = _pre.transform(X_te)
        model.fit(Xtr_, y_tr, epochs=40, batch_size=32, verbose=0)
        preds = (model.predict(Xte_, verbose=0).ravel() > 0.5).astype(int)
        f1 = f1_score(y_te, preds); acc = accuracy_score(y_te, preds)

        class AnnPipe:
            """Minimal wrapper that mimics a Pipeline for the app."""
            def __init__(self, pre, net):
                self.pre, self.net = pre, net
                self.named_steps = {"pre": pre, "clf": net}
            def predict(self, X):
                X_ = self.pre.transform(X)
                return (self.net.predict(X_, verbose=0).ravel() > 0.5).astype(int)
            def predict_proba(self, X):
                X_ = self.pre.transform(X)
                p = self.net.predict(X_, verbose=0).ravel()
                return np.vstack([1-p, p]).T

        est = AnnPipe(_pre, model)
        results.append(("ann", f1, acc, None)); artifacts["ann"] = est

    # ---------------- choose best & save ----------------
    res_df = pd.DataFrame(results, columns=["model","f1","accuracy","best_params"]).sort_values("f1", ascending=False)
    print("\n=== Results (sorted by F1) ===")
    print(res_df.to_string(index=False))

    best_name = res_df.iloc[0]["model"]
    best_est = artifacts[best_name]
    print(f"\nWinner: {best_name}")

    # Save per-model artifacts (optional)
    for name, est in artifacts.items():
        try:
            joblib.dump(est, models_dir / f"pipeline_{name}.pkl", compress=3)
        except Exception as e:
            warnings.warn(f"Could not save {name}: {e}")

    # Save BEST pipeline for the app
    final_path = models_dir / "best_pipeline.pkl"
    joblib.dump(best_est, final_path, compress=5)

    # Metrics
    res_df.to_csv(models_dir / "metrics.csv", index=False)
    (models_dir / "best_name.txt").write_text(str(best_name))

    print(f"\nSaved: {final_path}")
    print("Also wrote: models/metrics.csv and models/best_name.txt")

if __name__ == "__main__":
    main()