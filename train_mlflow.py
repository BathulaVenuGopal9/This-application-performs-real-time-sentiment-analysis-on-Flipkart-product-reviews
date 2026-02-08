# ===============================
# IMPORTS
# ===============================
import pandas as pd
import numpy as np
import re
import os
import tempfile
import joblib
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
# MAIN TRAIN FUNCTION
# ===============================
def train_models():

    # LOAD DATA
    df = pd.read_csv("cleaned_flipkart.csv")
    df["clean_text"] = df["reviewed_text"].apply(clean_text)

    X = df["clean_text"]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MODELS
    model_map = {
        "nb": MultinomialNB(),
        "lr": LogisticRegression(max_iter=1000, random_state=42),
        "dt": DecisionTreeClassifier(random_state=42),
        "svc": SVC(probability=True, random_state=42),
        "rf": RandomForestClassifier(random_state=42),
        "xgb": XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42),
    }

    # ===============================
    # OPTUNA OBJECTIVE (NESTED RUN)
    # ===============================
    def objective(trial):

        with mlflow.start_run(nested=True):

            model_name = trial.suggest_categorical("model", list(model_map.keys()))
            model = model_map[model_name]

            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(stop_words="english")),
                ("model", model),
            ])

            score = cross_val_score(
                pipeline, X_train, y_train, cv=3, scoring="f1_macro"
            ).mean()

            # Log for Hyperparameter charts
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("trial_model", model_name)
            mlflow.log_metric("trial_f1_score", score)

            return score

    # ===============================
    # RUN OPTUNA
    # ===============================
    mlflow.set_experiment("Flipkart_Sentiment_Optuna")

    with mlflow.start_run(run_name="Optuna_Hyperparameter_Search"):

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        best_model_name = study.best_trial.params["model"]
        best_model = model_map[best_model_name]

        print("Best Model:", best_model_name)

        # FINAL TRAIN
        best_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("model", best_model),
        ])

        best_pipeline.fit(X_train, y_train)
        preds = best_pipeline.predict(X_test)

        final_f1 = f1_score(y_test, preds, average="macro")
        accuracy = (preds == y_test).mean()

        print("Final F1:", final_f1)
        print("Accuracy:", accuracy)

        # LOG FINAL METRICS
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("Final_F1", final_f1)
        mlflow.log_metric("Accuracy", accuracy)

        # MODEL SIZE
        tmp = tempfile.NamedTemporaryFile(delete=False)
        joblib.dump(best_pipeline, tmp.name)
        mlflow.log_metric("model_size_bytes", os.path.getsize(tmp.name))

        # TAGS (Requirement)
        mlflow.set_tag("project", "Flipkart Sentiment Analysis")
        mlflow.set_tag("algorithm", best_model_name)
        mlflow.set_tag("metric", "F1 Score")
        mlflow.set_tag("stage", "Production")

        # LOG & REGISTER MODEL
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="best_model",
            registered_model_name="Flipkart_Sentiment_Model"
        )

    return best_model_name, best_pipeline, final_f1


# RUN DIRECTLY
if __name__ == "__main__":
    train_models()





