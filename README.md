# XAI
Bank Churn Prediction Using Explainable Artificial Intelligence 
This project aims to predict customer churn for ABC Multistate Bank using Explainable AI

Churn Prediction

Overview
This project is an ML-based interactive web application built with Streamlit that predicts whether a bank customer is likely to churn (leave) based on various customer attributes. The prediction model used in this application is powered by a Random Forest Classifier, XGBoost Classifier, and Decision Tree classifier. It is complemented by eXplainable Artificial Intelligence (XAI) using SHAP (SHapley Additive explanations) to & Lime (Local Interpretable Model-agnostic Explanations) to provide insights into the model's predictions, making it transparent and interpretable.The dataset used for this project contains various features that may influence customer churn, such as credit score, age, tenure, balance, product usage, credit card status, active membership, estimated salary, and more. The target variable, "churn," indicates whether a customer has left the bank during a specific period (1 if churned, 0 if not).

Getting Started
To run the project, follow these steps:

Clone the repository:
https://github.com/Omkar-Rajkumar-Khade/Customer_churn_prediction_using_ann_using_XAI.git
Install the required libraries:
pip install pandas numpy scikit-learn shap lime matplotlib streamlit
Open the Jupyter Notebook CCP_using_SML.ipynb using Jupyter Notebook or any compatible environment.

Open the terminal or command prompt and navigate to the repository directory.

Run the Streamlit app: streamlit run streamlit_app.py

The app will open in your default web browser, allowing you to input feature values and see churn predictions.

Note: Please update the file paths if necessary and ensure that the required libraries are installed.

Technologies Used:
Python
Streamlit
scikit-learn ( machine learning)
pickle
SHAP (SHapley Additive exPlanations)
Lime (Local Interpretable Model-agnostic Explanations)
Usage
app.py

Use the sliders and input fields in the web app to input customer data such as credit score, age, tenure, balance, products, credit card status, active membership, and estimated salary.

Click the "Predict" button to obtain the model's prediction for customer churn.

The application will display whether the customer is likely to churn or not, along with the churn probability.

CCP_using_SML.ipynb

Explore the SHAP summary plot to understand the impact of each feature on the model's prediction, making it transparent and interpretable.
Dataset
The dataset used for this project contains the following columns: Dataset Download Link: https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

customer_id: Unused variable.

credit_score: Used as an input.

country: Unused variable.

gender: Unused variable.

age: Used as an input.

tenure: Used as an input.

balance: Used as an input.

products_number: Used as an input.

credit_card: Used as an input.

active_member: Used as an input.

estimated_salary: Used as an input.

churn: Target variable. 1 if the client has left the bank during some period or 0 if he/she has not.

Repository Files
The repository contains the following files:

dataset folder contains the Bank Customer Churn Prediction.csv dataset used in the project.

app.py is the streamlit application file that defines the API endpoints and loads the saved model.

models is a folder that contains the serialized machine learning models that is used for prediction.

CCP_using_SML.ipynb: Jupyter Notebook containing the code for data loading, preprocessing, model building, training, and evaluation.

README.md: Project documentation and instructions.

Code Structure
The code performs the following steps:

Data Preprocessing:

Reads the dataset using Pandas.
Drops irrelevant columns ('country' and 'gender').
Visualizes the correlation matrix and numeric feature distributions.
Data Split:

Splits the data into training and testing sets.
Data Scaling:

Standardizes the features using StandardScaler.
Model Building:

Creates three classifiers: Random Forest, Decision Tree, and XGBoost.
Trains the models on the training data.
Model Evaluation:

Predicts on the test data.
Calculates and prints accuracy for each model.
Generates a confusion matrix and heatmap for the Random Forest model.
Model Interpretability:

Uses SHAP (SHapley Additive exPlanations) to explain model predictions.
Uses LIME (Local Interpretable Model-agnostic Explanations) to explain a specific prediction.
Model Serialization:

Saves the trained Random Forest model to a .pkl file.
Frontend :

Interactive web application frontend built with Streamlit
Data Preprocessing & Exploratory Data Analysis
The code starts by importing necessary libraries such as NumPy, pandas, matplotlib, seaborn, and scikit-learn.
It loads the dataset using pandas and drops irrelevant columns ('country' and 'gender').
Data is split into training and testing sets, and feature scaling is applied using StandardScaler.
The correlation matrix of the dataset is visualized using a heatmap.
The distribution of numeric features is visualized through histograms.
Model Building
Three machine learning models are employed:

Random Forest Classifier: A Random Forest classifier is created and trained on the scaled training data.
Decision Tree Classifier: A Decision Tree classifier is created and trained on the scaled training data.
XGBoost Classifier: An XGBoost classifier is created and trained on the scaled training data.
Model Evaluation :
Accuracy is calculated for each model using the test data.
Confusion matrices are generated and visualized using seaborn.
The project employs different classification algorithms, and here are their accuracy scores:

Random Forest Classifier: 0.8565 (86%)
Decision Tree Classifier: 0.7875(79%)
XGBoost Classifier: 0.8465 (85%) These accuracy scores reflect the performance of the respective models in predicting customer churn.
Explainable AI
we have added explanations for model predictions using SHAP and LIME techniques for a selected data point.

SHAP (Shapley Additive exPlanations) is a model-agnostic method for explaining the predictions of any machine learning model. It works by calculating the contribution of each feature to the prediction, taking into account all possible combinations of features.

SHAP Explanation
The SHAP Explanation image shows the contribution of each feature to the prediction of the Random Forest Classifier for the selected instance. The features are represented by horizontal bars, and the length of a bar represents the magnitude of its contribution. The color of a bar indicates whether the contribution is positive (red) or negative (blue).

The following is an explanation of the image, with reference to the code provided:

The age feature has a small negative contribution to the prediction, which means that it slightly decreases the probability of the customer churning.
The tenure feature has a small positive contribution to the prediction, which means that it slightly increases the probability of the customer churning.
The balance feature has a small negative contribution to the prediction, which means that it slightly decreases the probability of the customer churning.
The products_number feature has a small negative contribution to the prediction, which means that it slightly decreases the probability of the customer churning.
The estimated_salary feature has a small positive contribution to the prediction, which means that it slightly increases the probability of the customer churning.
LIME (Local Interpretable Model-Agnostic Explanations) is another model-agnostic method for explaining the predictions of any machine learning model. It works by creating a local linear model that approximates the predictions of the original model around a specific instance.

Lime Explanation

The LIME Explanation image shows the contribution of each feature to the prediction of the Random Forest Classifier for the selected instance. The features are represented by bars, and the length of a bar represents the magnitude of its contribution. The color of a bar indicates whether the contribution is positive (red) or negative (blue).

The following is an explanation of the image, with reference to the code provided:
The credit_score feature has the largest positive contribution to the prediction, which means that it is the most important feature in explaining why the model predicted that the customer will churn.
The age feature has a small negative contribution to the prediction, which means that it slightly decreases the probability of the customer churning.
The tenure feature has a small positive contribution to the prediction, which means that it slightly increases the probability of the customer churning.
The balance feature has a small negative contribution to the prediction, which means that it slightly decreases the probability of the customer churning.
The products_number feature has a small negative contribution to the prediction, which means that it slightly decreases the probability of the customer churning. T6) he estimated_salary feature has a small positive contribution to the prediction, which means that it slightly increases the probability of the customer churning. Overall, the LIME Explanation image shows that the Random Forest Classifier is predicting that the customer will churn because of their low credit score and high age. However, the model is also taking into account other factors, such as the customer's tenure, balance, number of products, and estimated salary.
Save Model:
The Random Forest model is saved using pickle and can be used for deployment in a real-world application

Streamlit Web App for Predictions
In addition to the machine learning model, this repository includes a Streamlit web app for making real-time predictions on customer churn. The app allows users to input customer information and receive predictions on whether a customer is likely to churn or not.
