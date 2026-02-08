from prefect import task, flow
import pandas as pd
import pickle
from train_mlflow import train_models


@task
def load_data(path="cleaned_flipkart.csv"):
    df = pd.read_csv(path)
    print("Data Loaded:", df.shape)
    return df


@task
def train():
    best_model_name, best_pipeline, best_f1 = train_models()
    print("Best Model:", best_model_name)
    print("Best F1:", best_f1)
    return best_model_name, best_pipeline, best_f1


@task
def save_model(best_pipeline):
    pickle.dump(best_pipeline, open("best_model.pkl", "wb"))
    print("Model saved as best_model.pkl")


@flow(name="Flipkart_MLOps_Pipeline")
def mlops_pipeline():

    load_data()
    best_model_name, best_pipeline, best_f1 = train()
    save_model(best_pipeline)

    print("\nPIPELINE COMPLETED")
    print("Best Model:", best_model_name)
    print("Best F1:", best_f1)


if __name__ == "__main__":
    mlops_pipeline()




