import pandas as pd
import numpy as np
import re
import string
import mlflow
import mlflow.sklearn
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ===============================
# TEXT CLEANING
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


# ===============================
# MAIN TRAIN FUNCTION (IMPORTANT)
# ===============================
def train_models():

    # ===============================
    # LOAD DATA
    # ===============================
    df = pd.read_csv("cleaned_flipkart.csv")

    # Use correct column names
    df["clean_text"] = df["reviewed_text"].apply(clean_text)

    X = df["clean_text"]
    y = df["Target"]

    # ===============================
    # SPLIT
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ===============================
    # MODEL MAP
    # ===============================
    model_map = {
        "nb": MultinomialNB(),
        "lr": LogisticRegression(max_iter=1000, random_state=42),
        "dt": DecisionTreeClassifier(random_state=42),
        "svc": SVC(probability=True, random_state=42),
        "rf": RandomForestClassifier(random_state=42),
        "xgb": XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        ),
    }

    # ===============================
    # OPTUNA OBJECTIVE
    # ===============================
    def objective(trial):

        model_name = trial.suggest_categorical("model", list(model_map.keys()))
        model = model_map[model_name]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("model", model),
        ])

        score = cross_val_score(
            pipeline, X_train, y_train, cv=3, scoring="f1_macro"
        ).mean()

        return score

    # ===============================
    # RUN OPTUNA
    # ===============================
    mlflow.set_experiment("Flipkart_Sentiment_Optuna")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_trial.params
    print("Best Params:", best_params)

    # ===============================
    # FINAL TRAINING
    # ===============================
    best_model_name = best_params["model"]
    best_model = model_map[best_model_name]

    best_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("model", best_model),
    ])

    best_pipeline.fit(X_train, y_train)
    preds = best_pipeline.predict(X_test)

    final_f1 = f1_score(y_test, preds, average="macro")
    print("Final F1:", final_f1)

    # ===============================
    # LOG TO MLFLOW + REGISTER MODEL
    # ===============================
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Final_Best_Model"):

        mlflow.log_metric("Final_F1", final_f1)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="best_model",
            registered_model_name="Flipkart_Sentiment_Model"
        )

    print("Model Registered in MLflow")

    return best_model_name, best_pipeline, final_f1



