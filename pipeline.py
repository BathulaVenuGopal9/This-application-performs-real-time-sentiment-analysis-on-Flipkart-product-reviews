from prefect import task, flow
import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from train_mlflow import train_models


# TASK 1 — LOAD DATA (OPTIONAL, training already loads data)
@task
def load_data(path="cleaned_flipkart.csv"):
    df = pd.read_csv(path)
    print("Data Loaded:", df.shape)
    return df


# TASK 2 — TRAIN MODEL
@task
def train():
    best_model_name, best_pipeline, best_f1 = train_models()
    print("Best Model:", best_model_name)
    print("Best F1:", best_f1)
    return best_model_name, best_pipeline, best_f1


# TASK 3 — LOG & REGISTER MODEL
@task
def log_and_register(best_model_name, best_pipeline, best_f1):

    mlflow.set_experiment("MLOPS_TASK2")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Prefect_Final_Model"):

        mlflow.log_metric("Final_F1", best_f1)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="best_model",
            registered_model_name="Flipkart_Sentiment_Model"
        )

        print("Model logged & registered in MLflow")

    pickle.dump(best_pipeline, open("best_model.pkl", "wb"))
    print("Model saved locally as best_model.pkl")


# MAIN FLOW
@flow(name="Flipkart_MLOps_Pipeline")
def mlops_pipeline():

    load_data()  # optional (just for log)
    best_model_name, best_pipeline, best_f1 = train()
    log_and_register(best_model_name, best_pipeline, best_f1)

    print("\nPIPELINE COMPLETED")
    print("Best Model:", best_model_name)
    print("Best F1:", best_f1)


if __name__ == "__main__":
    mlops_pipeline()

