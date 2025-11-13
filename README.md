# Customer Churn Prediction for Telecommunications

> A machine learning project focused on identifying telecom customers who are likely to churn, using Python, scikit-learn, and pandas.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Status](https://img.shields.io/badge/status-complete-green.svg)

---

##  Table of Contents

* [Project Overview](#-project-overview)
* [Business Problem](#-business-problem)
* [Data](#-data)
* [Methodology](#-methodology)
* [Results](#-results)
* [Installation](#-installation)
* [Usage](#-usage)
* [License](#-license)
* [Acknowledgements](#-acknowledgements)

---

##  Project Overview

This project implements a complete machine learning pipeline to predict customer churn. The goal is to build a model that can accurately identify customers at high risk of leaving the service. By identifying these customers, the business can take proactive measures, such as offering special discounts or incentives, to retain them.

This project uses Python and popular data science libraries including:
* **Pandas:** For data manipulation and cleaning.
* **Scikit-learn:** For feature engineering, model training, and evaluation.

The final output is a trained Random Forest classifier that achieves an **ROC-AUC score of 0.81** on the test set, effectively balancing the identification of churning customers.

---

##  Business Problem

Customer churn (the rate at which customers stop doing business with an entity) is a critical metric for subscription-based businesses like telecommunications. It is significantly more expensive to acquire a new customer than to retain an existing one.

This project addresses this problem by answering the following questions:
1.  What are the key factors driving customer churn?
2.  Can we build a model to predict which customers are at high risk of churning?
3.  What recommendations can be made to the business to help reduce churn?

---

##  Data

The dataset used for this project is the **Telco Customer Churn** dataset, originally from Kaggle.

* **Source:** [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Description:** The dataset contains 7043 customer records with 21 attributes. Each record represents a unique customer and includes demographic info, account details, services they subscribe to, and whether they churned in the last quarter.
* **Key Features:**
    * `gender`, `SeniorCitizen`, `Partner`, `Dependents` (Customer Demographics)
    * `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` (Account Information)
    * `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, etc. (Services)
* **Target Variable:** `Churn` (Yes/No)
* **Data Cleaning:**
    * Converted `TotalCharges` from an object to a numeric type, filling 11 missing values (for new customers) with 0.
    * Converted `SeniorCitizen` from 0/1 to a more descriptive 'No'/'Yes' for consistency.
    * Standardized categorical feature values for easier encoding.

---

## 🛠️ Methodology

The project was structured in several key phases:

1.  **Feature Engineering & Preprocessing:**
    * **Categorical Encoding:** Used One-Hot Encoding for categorical features with multiple values (e.g., `Contract`, `PaymentMethod`).
    * **Binary Encoding:** Converted binary features (e.g., `Partner`, `Dependents`) to 0/1.
    * **Scaling:** Applied `StandardScaler` to numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) to normalize their range.

2.  **Model Selection & Training:**
    * **Train/Test Split:** Split the data into 80% training and 20% testing sets.
    * **Handling Imbalance:** Applied inbuilt class weight parameter and set the value to balanced to notify the model that the dataset is imbalanced.
    * **Models Tested:**
        1.  Logistic Regression (as a baseline)
        2.  Decision Tree
        3.  Random Forest Classifier
        4.  XGBoost Classifier
    * **Evaluation Metric:** Due to the class imbalance, **ROC-AUC** was chosen as the primary metric. **Recall** was also heavily considered, as it's important to find as many *actual* churners as possible (minimizing false negatives).

---

##  Results

The models were evaluated on the unseen test set. The Random Forest Classifier provided the best balance of performance and interpretability.

**Model Performance Comparison (Test Set):**

| Model | Precision (Churn=Yes) | Recall (Churn=Yes) | F1-Score (Churn=Yes) | ROC-AUC |
| **Random Forest** | **0.62** | **0.61** | **0.62** | **0.80** |


**Conclusion:**
The final model (Random Forest) can effectively identify customers at high risk of churn (ROC-AUC 0.80).

**Business Recommendations:**
1.  **Target Month-to-Month Customers:** Proactively offer incentives for them to switch to One-year or Two-year contracts.
2.  **Bundle Key Services:** Promote bundles that include `OnlineSecurity` and `TechSupport`, especially for new customers, as these services are linked to higher retention.
3.  **Review High-Spend/Low-Tenure Segment:** Customers with high monthly charges and low tenure are a high-risk group that may need targeted retention offers.

---

## ⚙️ Installation

To run this project locally, please follow these steps.

1.  Clone the repository:
    ```sh
    git clone [https://github.com/](https://github.com/)[your-username]/[your-project-name].git
    cd [your-project-name]
    ```
2.  Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

---

##
Usage

The project is divided into several files:

* `1_Data_Cleaning_and_EDA.ipynb`: Jupyter notebook for data loading, cleaning.
* `2_Model_Training_and_Evaluation.ipynb`: Jupyter notebook for feature preprocessing, model training, and performance evaluation.
* `models/`: This directory contains the saved (pickled) final model and preprocessor.


jupyter notebook

