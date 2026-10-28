"""
Wine Quality: steg 1 og 2
  Steg 1: Klassifisering i tre kategorier (lav / middels / høy kvalitet)
  Steg 2: Baseline + tre modeller med enkel tuning, sammenlignet på samme testsett
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
LABELS = ["lav", "middels", "høy"]  # 0, 1, 2


def quality_to_class(q: int) -> int:
    """Steg 1: 3-5 -> 0 (lav), 6 -> 1 (middels), 7-9 -> 2 (høy)."""
    if q <= 5:
        return 0
    if q == 6:
        return 1
    return 2


def load_data(red_path: str, white_path: str) -> pd.DataFrame:
    red = pd.read_csv(red_path, sep=None, engine="python")
    white = pd.read_csv(white_path, sep=None, engine="python")
    red["color"] = 0  # 0 = rød
    white["color"] = 1  # 1 = hvit
    df = pd.concat([red, white], ignore_index=True)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def main(args):
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # ---------- Data ----------
    df = load_data(args.red, args.white)
    print(f"Rader før duplikatfjerning: {len(df)}")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rader etter duplikatfjerning: {len(df)}")

    df["quality_class"] = df["quality"].apply(quality_to_class)
    print("\nKlassefordeling (andel):")
    print(df["quality_class"].map(dict(enumerate(LABELS))).value_counts(normalize=True).round(3))

    feature_cols = [c for c in df.columns if c not in ("quality", "quality_class")]
    X = df[feature_cols]
    y = df["quality_class"]

    # Duplikater er fjernet, så en vanlig stratifisert deling er trygg.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"\nTrening: {len(X_train)} rader, test: {len(X_test)} rader")

    # ---------- Modeller (steg 2) ----------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    models = {
        # Baseline 1: gjetter alltid den vanligste klassen
        "Dummy (vanligste klasse)": (DummyClassifier(strategy="most_frequent"), None),
        # Baseline 2: enkel og forklarbar modell
        "Logistisk regresjon": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=SEED, n_jobs=-1
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 3],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
    }

    results = []
    fitted = {}
    for name, (model, grid) in models.items():
        print(f"\n=== {name} ===")
        if grid:
            search = GridSearchCV(model, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            print("Beste parametre:", search.best_params_)
            cv_f1 = search.best_score_
            params = search.best_params_
        else:
            best = model.fit(X_train, y_train)
            cv_f1 = float("nan")
            params = {}

        pred = best.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        print(f"CV macro-F1 (trening): {cv_f1:.3f}")
        print(f"Test accuracy: {acc:.3f} | Test macro-F1: {f1:.3f}")
        print(classification_report(y_test, pred, target_names=LABELS, zero_division=0))

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=LABELS, ax=ax, cmap="Blues"
        )
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{name.split()[0].lower()}.png", dpi=150)
        plt.close(fig)

        results.append(
            {
                "modell": name,
                "cv_macro_f1": round(cv_f1, 3),
                "test_accuracy": round(acc, 3),
                "test_macro_f1": round(f1, 3),
                "parametre": json.dumps(params),
            }
        )
        fitted[name] = best

    # ---------- Sammenligning ----------
    res = pd.DataFrame(results).sort_values("test_macro_f1", ascending=False)
    print("\n=== Sammenligning (sortert på test macro-F1) ===")
    print(res.drop(columns="parametre").to_string(index=False))
    res.to_csv(out / "model_comparison.csv", index=False)

    best_name = res.iloc[0]["modell"]
    print(f"\nBeste modell: {best_name}")
    joblib.dump(
        {"model": fitted[best_name], "features": feature_cols, "labels": LABELS},
        out / "best_model.joblib",
    )

    # Min/maks/snitt per feature, nyttig for skyveknappene i nettsiden senere
    stats = X.describe().loc[["min", "mean", "max"]].round(4)
    stats.to_json(out / "feature_ranges.json")
    print(f"Lagret modell og figurer i mappen '{out}/'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--red", default="data/winequality-red.csv")
    p.add_argument("--white", default="data/winequality-white.csv")
    p.add_argument("--out", default="output")
    main(p.parse_args())