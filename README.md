# Flipkart Sentiment Analysis — End-to-End MLOps Project

## Project Overview
This project is a complete End-to-End Machine Learning + MLOps pipeline for predicting customer sentiment from Flipkart product reviews.
It includes data preprocessing, model selection using Optuna, experiment tracking with MLflow, orchestration using Prefect, and deployment via Streamlit Cloud.
The system automatically selects the best performing model and serves real-time sentiment predictions through a simple web interface.

## Objective
To build a scalable, reproducible, and production-ready sentiment analysis system that:
Cleans and processes real-world review text
Automatically selects the best ML model using Optuna
Tracks experiments and registers models using MLflow
Automates pipeline using Prefect
Deploys prediction service using Streamlit Cloud

## Machine Learning Workflow
Data Preprocessing
Text cleaning
Lowercasing, punctuation removal, regex cleaning
TF-IDF vectorization
Model Training & Selection
Models tested:
Naive Bayes
Logistic Regression
Decision Tree
Support Vector Machine
Random Forest
XGBoost
Hyperparameter & model selection using Optuna
Evaluation
Metric used: Macro F1 Score
Best model automatically selected
Experiment Tracking
Logged with MLflow
Model registered in MLflow Model Registry
Pipeline Automation
Orchestrated using Prefect
Fully reproducible pipeline
Deployment
Streamlit Web App
Real-time sentiment prediction

## Project Architecture
Copy code

Data → Preprocessing → Model Selection (Optuna) → MLflow Tracking
      → Prefect Pipeline → Model Registry → Streamlit Deployment
      
## Model Performance
Best Model: Auto-selected (SVC / XGBoost depending on run)
Evaluation Metric: Macro F1 Score
Production-ready model registered in MLflow

## Tech Stack
Category
Tools Used
Programming
Python
ML Libraries
Scikit-learn, XGBoost
Hyperparameter Tuning
Optuna
Experiment Tracking
MLflow
Pipeline Orchestration
Prefect
Deployment
Streamlit Cloud
NLP
NLTK, TF-IDF
Data Handling
Pandas, NumPy

## Project Structure
Copy code

mlops/
│
├── app.py                  # Streamlit prediction app
├── train_mlflow.py         # Model training + MLflow logging
├── pipeline.py             # Prefect orchestration pipeline
├── best_model.pkl          # Trained production model
├── cleaned_flipkart.csv    # Dataset
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

## Running Locally
1. Install dependencies
Copy code

pip install -r requirements.txt
2. Run Streamlit app
Copy code

streamlit run app.py
☁️ Deployment — Streamlit Cloud
The app is deployed using Streamlit Community Cloud directly from GitHub.
Steps followed:
Push project to GitHub
Connect repository in Streamlit Cloud
Select app.py as entry file
Auto-install dependencies from requirements.txt
App deployed successfully 

## MLOps Features Implemented
Experiment tracking with MLflow
Model versioning & registry
Automated training pipeline (Prefect)
Reproducible workflow
Production model deployment
End-to-End ML lifecycle

## Future Improvements
Add Deep Learning (LSTM / BERT)
Add API deployment (FastAPI)
CI/CD automation (GitHub Actions)
Docker containerization
Model monitoring & drift detection
Real-time streaming data pipeline
# Author
Bathula Venu Gopal
Machine Learning & MLOps Enthusiast
End-to-End ML | NLP | Deployment | Automation
