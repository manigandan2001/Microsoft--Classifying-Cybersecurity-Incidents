# Microsoft--Classifying-Cybersecurity-Incidents
Cybersecurity Capstone Project
Overview
This project aims to build a classification model to predict incident grades based on various features. The dataset used includes categorical and numerical data related to incidents, and the task is to classify the severity or grade of incidents into three categories: 0, 1, and 2. The project leverages multiple models using a Voting Classifier to combine their predictions and improve overall accuracy.

Files
Cybersecurity_Capstone.ipynb: The main Jupyter notebook containing all the preprocessing, model training, and evaluation steps.
README.md: The document describing the purpose of the project, the methodology, the structure of the notebook, and how to run it.
Project Structure
1. Data Preprocessing
Data Loading: The dataset is loaded into a Pandas DataFrame.
Handling Missing Data: Any missing values in the dataset are addressed either by imputation or by dropping irrelevant columns.
Feature Engineering: Additional features are created based on existing data. For example, the hour is categorized into 'Morning', 'Afternoon', and 'Evening' based on the time of the incident.
Encoding Categorical Variables: Categorical columns are encoded using Label Encoding to transform them into numerical format for machine learning algorithms.
Scaling: Numerical features such as 'Year' are scaled to ensure uniformity across features.
2. Model Training
Resampling the Dataset: The dataset is resampled to balance the classes for the target variable, IncidentGrade.
Models Used: Several classification models are trained, including Random Forest, Logistic Regression, and Gradient Boosting. These are then combined into a Voting Classifier, which takes the majority vote from all models for the final prediction.
Cross-Validation: Cross-validation is used to tune the hyperparameters of each model.
3. Model Evaluation
Accuracy: The model's overall accuracy is calculated to evaluate its performance.
Confusion Matrix: The confusion matrix is used to visualize the performance of the classifier and identify misclassifications.
Classification Report: Precision, recall, and F1-score are computed for each class to provide a more detailed assessment of the model.
4. Final Model Evaluation on Test Data
The preprocessed test data is passed through the trained Voting Classifier.
The performance is evaluated using the following metrics:
Accuracy: How often the classifier predicts correctly.
Classification Report: A detailed breakdown of precision, recall, and F1-score for each class.
Confusion Matrix: Displays the true positives, false positives, true negatives, and false negatives for each class.
Results
Accuracy: The final model achieved an accuracy of 88.5% on the test data.
Classification Report: The precision, recall, and F1-score for each class are provided in detail.
Confusion Matrix: The confusion matrix highlights where the model performs well and where it misclassifies.
Example of Output Metrics:
Accuracy: 0.8851584091772597
Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.95      0.91   1752940
           1       0.88      0.81      0.84    902698
           2       0.91      0.86      0.88   1492354

    accuracy                           0.89   4147992
   macro avg       0.89      0.87      0.88   4147992
weighted avg       0.89      0.89      0.88   4147992

Confusion Matrix:
[[1660162   49435   43343]
 [  90953  730723   81022]
 [ 156675   54934 1280745]]
Prerequisites
Make sure you have the following libraries installed

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Conclusion
The Voting Classifier successfully predicted incident grades with an accuracy of 88.5%. The model shows strong performance in classifying most incidents.
