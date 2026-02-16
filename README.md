Medical Insurance Cost Prediction using XGBoost
1. Introduction

This project aims to predict individual medical insurance charges based on demographic and health-related attributes. The prediction model is built using the XGBoost Regressor, a high-performance gradient boosting algorithm known for its accuracy and efficiency.

The system demonstrates an end-to-end machine learning workflow including data preprocessing, exploratory analysis, feature engineering, model training, evaluation, and prediction.

2. Problem Statement

Insurance providers determine medical premiums based on multiple risk factors such as age, BMI, smoking habits, and number of dependents. Manual premium estimation can be inefficient and inconsistent.

This project builds a regression model that:

Learns patterns from historical insurance data

Predicts insurance charges for new individuals

Helps understand the impact of various features on pricing


3. Dataset Description

The dataset contains medical insurance records with the following features:

Age – Age of the individual

Sex – Gender

BMI – Body Mass Index

Children – Number of dependents covered

Smoker – Smoking status (Yes/No)

Region – Residential region

Charges – Medical insurance cost (Target Variable)

Target Variable: Charges

4. Exploratory Data Analysis

Exploratory analysis was performed to understand feature distributions and relationships with the target variable.

Key observations:

Smoking status has a strong positive correlation with insurance charges

BMI and age show a positive relationship with cost

Smokers incur significantly higher medical expenses compared to non-smokers

Regional differences have relatively lower impact compared to smoking and BMI

5. Data Preprocessing

The following preprocessing steps were applied:

Handling categorical variables using encoding techniques

Feature selection

Train-test split

Scaling (if applied)

Categorical features such as sex, smoker, and region were converted into numerical representations suitable for model training.

6. Model Used

XGBoost Regressor

The primary model used in this project is XGBoost (Extreme Gradient Boosting).

Reasons for choosing XGBoost:

High predictive performance

Built-in regularization to prevent overfitting

Handles non-linear relationships effectively

Efficient training and parallel computation

7. Model Evaluation

The model was evaluated using standard regression metrics:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Example (replace with actual values):

R² Score: 0.89
MAE: 2500
RMSE: 4200

The results indicate that the model is able to explain a high proportion of variance in insurance charges.

8. Project Structure

medical-insurance-cost-prediction/
│
├── data/
├── notebooks/
├── model/
├── app.py
├── requirements.txt
└── README.md

10. Installation and Setup

Clone the repository:

git clone https://github.com/KaranYadav09/Medical_Insurance_cost_prediction.git

Navigate to the project directory:

cd medical-insurance-cost-prediction

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

10. Usage

Provide input values such as age, BMI, smoking status, and number of children.

Submit the input to the model.

The system returns the predicted insurance cost.

11. Business Impact

Automates insurance premium estimation

Reduces manual underwriting effort

Enables data-driven decision-making

Helps identify high-risk individuals

12. Future Improvements

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV

Model explainability using SHAP

REST API integration

Cloud deployment (AWS, Azure, or GCP)

Integration with real-time insurance systems
