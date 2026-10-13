"""Career Switch Prediction (BRAC University CSE422, Group 17).

Predicts whether a professional will switch careers (will_change_career = 1)
from demographic, education and work-history features.

    python Career_Switch_Classification.py           # train and evaluate all models
    python Career_Switch_Classification.py --plots   # also show the exploratory plots
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA = Path(__file__).with_name("Career_Switch_Prediction_Dataset.csv")
TARGET = "will_change_career"
NUMERIC = ["city_development_index", "relevent_experience", "experience", "training_hours"]
CATEGORICAL = ["city", "gender", "enrolled_university", "education_level",
               "major_discipline", "company_size", "company_type", "last_new_job"]
SEED = 42


def load():
    df = pd.read_csv(DATA).drop(columns="enrollee_id")
    df["experience"] = pd.to_numeric(df["experience"].replace({"<1": 0.5, ">20": 21}))
    df["company_size"] = df["company_size"].replace("Oct-49", "10-49")  # spreadsheet turned 10-49 into a date
    df["relevent_experience"] = df["relevent_experience"].map(
        {"Has relevent experience": 1, "No relevent experience": 0})
    return df


def plots(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric = df.select_dtypes(include="number")
    numeric.hist(figsize=(8, 8), bins=20)
    plt.show()

    for col in CATEGORICAL:
        df[col].value_counts().sort_index().plot(kind="bar", title=f"Distribution of {col}", ylabel="count")
        plt.show()

    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.3)
    plt.show()

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for axis, method in zip(ax, ["pearson", "spearman", "kendall"]):
        corr = numeric.corr(method)[[TARGET]].sort_values(by=TARGET, ascending=False)
        sns.heatmap(corr, ax=axis, annot=True)
        axis.set_title(f"{method.title()} correlation with {TARGET}")
    plt.show()

    (df[TARGET].value_counts(normalize=True) * 100).plot(kind="bar", title="Class balance (%)")
    plt.show()


def preprocessor():
    # Fitted on the training split only (inside each pipeline), so the test set never leaks in.
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def neural_net_scores(X_train, y_train, X_test):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    prep = preprocessor().fit(X_train)
    X_train, X_test = prep.transform(X_train), prep.transform(X_test)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),  # one probability -> binary cross-entropy
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    return model.predict(X_test, verbose=0).ravel()


def evaluate(y_test, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    df = load()
    if "--plots" in sys.argv:
        plots(df)

    X, y = df.drop(columns=TARGET), df[TARGET]
    print(f"{len(df)} rows, {y.mean():.1%} will switch careers")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)

    models = {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    }
    results = {}
    for name, model in models.items():
        pipe = make_pipeline(preprocessor(), model).fit(X_train, y_train)
        results[name] = evaluate(y_test, pipe.predict_proba(X_test)[:, 1])
    results["Neural Network"] = evaluate(y_test, neural_net_scores(X_train, y_train, X_test))

    print("\nTest set (30%, stratified). Precision and recall are for the 'will switch' class.\n")
    print(pd.DataFrame(results).T.round(3).to_string())


if __name__ == "__main__":
    main()
