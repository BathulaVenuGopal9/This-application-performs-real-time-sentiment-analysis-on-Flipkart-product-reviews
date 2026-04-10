# Flipkart Sentiment Analysis — End-to-End MLOps Project (EC2 Deployment)
## Project Overview

This project is a complete End-to-End Machine Learning + MLOps pipeline for predicting customer sentiment from Flipkart product reviews. It demonstrates real-world production practices including automated training, experiment tracking, orchestration, and cloud deployment on AWS EC2.

The system automatically selects the best performing model and serves real-time sentiment predictions via a live web application hosted on AWS.

### Objective

To build a scalable, reproducible, and production-ready sentiment analysis system that:

Cleans and processes real-world review text
Automatically selects the best ML model using Optuna
Tracks experiments and registers models using MLflow
Automates pipeline using Prefect
Deploys the application on AWS EC2 (public access)

### Machine Learning Workflow
🔹 Data Preprocessing
Text cleaning (lowercasing, punctuation removal, regex)
Tokenization
TF-IDF Vectorization
🔹 Model Training & Selection

### Models evaluated:

Naive Bayes
Logistic Regression
Decision Tree
Support Vector Machine (SVC)
Random Forest
XGBoost

👉 Hyperparameter tuning using Optuna

👉 Best model selected using Macro F1 Score

### Experiment Tracking
MLflow used for:
Logging metrics
Tracking experiments
Model registry

### Pipeline Automation
Orchestrated using Prefect
Fully reproducible workflow

### Project Architecture
Data 
  → Preprocessing 
  → Model Selection (Optuna) 
  → MLflow Tracking 
  → Prefect Pipeline 
  → Model Registry 
  → Deployment (Streamlit + AWS EC2)
  
### Model Performance

✅ Best Model: Auto-selected (SVC / XGBoost)

✅ Evaluation Metric: Macro F1 Score

✅ Production-ready model registered in MLflow

### Tech Stack
Category	Tools Used
Programming	Python
ML Libraries	Scikit-learn, XGBoost
Hyperparameter Tuning	Optuna
Experiment Tracking	MLflow
Orchestration	Prefect
Deployment	Streamlit, AWS EC2
NLP	NLTK, TF-IDF
Data Handling	Pandas, NumPy

### Project Structure
mlops/
│
├── app.py                  # Streamlit prediction app
├── train_mlflow.py         # Model training + MLflow logging
├── pipeline.py             # Prefect pipeline
├── best_model.pkl          # Trained model
├── cleaned_flipkart.csv    # Dataset
├── requirements.txt        # Dependencies
└── README.md               # Documentation
☁️ Deployment — AWS EC2 (Production)

## Why EC2?

Unlike basic deployments, this project is deployed on AWS EC2, simulating real-world production environments.

### Steps Performed
1️⃣ Launch EC2 Instance
Ubuntu Server 24.04 LTS
Instance Type: t3.micro (Free Tier)

2️⃣ Configure Security Group

Opened ports:

22 → SSH access
8501 → Streamlit app

3️⃣ Connect to EC2
ssh -i mlops-flipkart.pem ubuntu@<PUBLIC-IP>

4️⃣ Upload Project Files
scp -i mlops-flipkart.pem -r mlops ubuntu@<PUBLIC-IP>:/home/ubuntu/

5️⃣ Install Dependencies
sudo apt update
pip install -r requirements.txt

6️⃣ Run Application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

7️⃣ Run in Background (Production Mode)
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

### Live Application

👉 Access via: http://16.52.175.80:8501

http://<EC2-PUBLIC-IP>:8501

### Key Learnings (Real DevOps Concepts)
Cloud deployment using AWS EC2
Security group configuration
Port management
Background process handling (nohup)
Debugging production issues
Handling multiple deployments

### Previous Deployment (Optional)

This project was initially deployed using:

Streamlit Community Cloud

Now upgraded to:
👉 Full EC2 Production Deployment

### MLOps Features Implemented
Experiment tracking (MLflow)
Model versioning & registry
Automated pipelines (Prefect)
Hyperparameter tuning (Optuna)
Cloud deployment (AWS EC2)
End-to-End ML lifecycle

### Future Improvements
Add Deep Learning (LSTM / BERT)
FastAPI deployment
CI/CD (GitHub Actions)
Docker containerization
Model monitoring & drift detection

### Author

Bathula Venu Gopal
Intern @ innomatics research labs |Machine Learning & MLOps Engineer


