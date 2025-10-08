🏆 **Kaggle Competition:** [Ultimate Customer Churn Prediction Challenge](https://www.kaggle.com/competitions/ultimate-customer-churn-prediction-challenge)
📊 **Colab Notebook:**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WSgfoS6q8sAJ79cdWoaA9WCY17BWH40U?usp=sharing)

# 🧠 Ultimate Customer Churn Prediction Challenge

This repository contains my Google Colab notebook submission for the **[Kaggle Ultimate Customer Churn Prediction Challenge](https://www.kaggle.com/competitions/ultimate-customer-churn-prediction-challenge/overview)**.  
The project’s goal is to build a reliable model to predict **customer churn** based on demographics, usage metrics, feedback, and service data.

---

## 📋 Competition Overview

Customer churn is a major challenge for businesses, as losing customers directly impacts revenue and growth.  
In this competition, your objective is to predict which customers are likely to leave using demographic, usage, and feedback data.  
The performance metric is **F1-score**, emphasizing a balance between precision and recall.

### 🎯 Objectives
- Build a churn prediction model using the provided datasets  
- Optimize for F1-score  
- Gain experience in feature engineering, data preprocessing, and model tuning  
- Compete for a high rank on the leaderboard  

---

## 📊 Dataset Description

You’ll work with structured datasets containing:
- **Demographics & Subscription** details  
- **Usage & Payment Behavior** metrics  
- **Feedback & Interaction History**  
- The target variable is `Churn` (1 = churn, 0 = no churn)

---

## 🧩 Implementation Walkthrough

### 1. Data Extraction & Loading  
- Extracted the provided ZIP file in Colab.  
- Loaded `train.csv` and `test.csv` with pandas.  

### 2. Data Cleaning & Preprocessing
- - Handled missing values and inconsistent entries such as `Gender`, `Subscription_Type`, and `Last_Interaction_Type` using mode.  
- Encoded categories:  
  - `Gender` → {Male: 0, Female: 1}  
  - `Subscription_Type` → {Basic: 0, Premium: 1}  
  - `Last_Interaction_Type` → {Neutral: 0, Negative: 1, Positive: 2}  
- Left `Location` as categorical to OneHotEncode later. 
- Balanced the dataset using **SMOTE** to handle class imbalance.

### 3. Feature & Target Split  
- Defined `X` (features) by dropping `Customer_ID` and `Churn`, and `y` as `Churn`.  
- Performed an 80/20 train–test split (random_state = 42).

### 4. Preprocessing Pipeline  
- **Numerical features** scaled with `MinMaxScaler`.  
- **Location** feature transformed with `OneHotEncoder`.  
- Used a `ColumnTransformer` to combine these.  
- Ensured train and test transformations use the same pipeline.

### 5. Handle Class Imbalance  
- Applied **SMOTE** to generate synthetic samples for minority class in training data.

### 6. Model Training  
- Trained an **AdaBoostClassifier** with parameters:
  - `n_estimators = 700`  
  - `learning_rate = 0.5`  
- Evaluated using classification report metrics.

### 7. Hyperparameter Tuning  
- Defined a parameter grid for `n_estimators` and `learning_rate`.  
- Used `GridSearchCV` (5-fold CV) optimizing **F1-score**.  
- Selected the best estimator for final predictions.

### 8. Model Evaluation  
- Evaluated on hold-out test set using:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  

### 9. Submission Creation & Kaggle Integration  
- Processed `test.csv` identically to training data.  
- Predicted churn probabilities with the final model.  
- Built `submission.csv` with:

Customer_ID,Churn_Probability
1001,0.24
1002,0.92
...

- Submitted automatically via Kaggle API using Colab integration.

<h2> 📈 Results </h2>
Metric	Score
Best Cross-Validation F1-score	0.87 (example, replace with your actual)
Leaderboard Rank	🏆 #43 out of 86
Best Model	AdaBoost (optimized with GridSearchCV + SMOTE)

<h2> Key insights: </h2>

- Subscription type, last interaction sentiment, and location strongly influence churn.

- SMOTE balancing improved minority class recall significantly.

- AdaBoost achieved a better bias-variance tradeoff than logistic regression or random forests.

<h2>🧠 Technologies Used</h2>

- Languages: Python

- Libraries: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, matplotlib

- Environment: Google Colab

- APIs: Kaggle API for automated submission

<h2>🚀 Future Improvements</h2>

- Implement stacking with AdaBoost + LightGBM for improved robustness.

- Visualize feature importance using SHAP values.

- Deploy churn prediction as an interactive web dashboard.
