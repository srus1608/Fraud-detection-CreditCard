# Fraud Detection of credit card using Machine Learning

## Overview

This project implements a fraud detection system using machine learning techniques, specifically Logistic Regression. The goal is to identify fraudulent transactions from a dataset of credit card transactions.

## Features

1. **Data Preprocessing:** Handling missing values, data normalization, and feature engineering.

2.  **Model Training:** Logistic Regression to classify transactions as fraudulent or genuine.
 
3. **Evaluation:** Model performance evaluation using metrics like accuracy, precision, recall, and F1-score.

4.  **Visualization:** Insights into the dataset and model performance.

## Dataset

The dataset used in this project is a collection of credit card transactions. You can download the dataset from the following link:

- [Download Dataset](https://www.kaggle.com/datasets/aniruddhachoudhury/creditcard-fraud-detection?resource=download))

**Note:** Ensure that you have the dataset in the correct format and location as specified in the code.

## Installation

1. **Clone the Repository:**

   
   git clone https://github.com/srus1608/Fraud-detection-CreditCard.git
   cd Fraud-detection-CreditCard
2.Install Required Libraries:

pip install numpy pandas scikit-learn matplotlib seaborn

Certainly! Here's a template for a README file for a fraud detection project using machine learning and Logistic Regression. You can adjust the placeholders and add more details as needed:

markdown
Copy code
# Fraud Detection using Machine Learning

## Overview

This project implements a fraud detection system using machine learning techniques, specifically Logistic Regression. The goal is to identify fraudulent transactions from a dataset of credit card transactions.

## Features

- **Data Preprocessing:** Handling missing values, data normalization, and feature engineering.
- **Model Training:** Logistic Regression to classify transactions as fraudulent or genuine.
- **Evaluation:** Model performance evaluation using metrics like accuracy, precision, recall, and F1-score.
- **Visualization:** Insights into the dataset and model performance.

## Dataset

The dataset used in this project is a collection of credit card transactions. You can download the dataset from the following link:

- [Download Dataset](<INSERT_DATASET_LINK_HERE>)

**Note:** Ensure that you have the dataset in the correct format and location as specified in the code.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/srus1608/Fraud-detection-CreditCard.git
   cd Fraud-detection-CreditCard
2. Install Required Libraries:

Make sure you have Python installed. Then, install the required libraries using pip:


pip install -r requirements.txt

If requirements.txt is not available, install libraries individually:

pip install numpy pandas scikit-learn matplotlib seaborn

### Usage
Load the Dataset:

Ensure that the dataset is in the correct directory or update the path in the code:

import pandas as pd
data = pd.read_csv('path_to_your_dataset.csv')

Run the Notebook:

Open the Jupyter notebook:

jupyter notebook Creditcard_fraud_detection_(1).ipynb

## Results
The Logistic Regression model is evaluated based on various metrics. Key results include:

Accuracy: 99.86%

Precision: The proportion of true positives out of all positive predictions.

