# Loan Approval Prediction Analysis

## Overview

This Python script analyzes a dataset from Kaggle for loan approval prediction. It employs exploratory data analysis (EDA) techniques, visualizes key insights, and utilizes machine learning to predict loan approval status based on various features.

## Prerequisites

1. **Clone Repository:**
   - Clone this repository to your local machine:

     ```bash
     git clone https://github.com/your-username/loan-approval-prediction.git
     ```

2. **Install Dependencies:**
   - Ensure you have the required libraries installed:

     ```bash
     pip install pandas numpy plotly scikit-learn
     ```

3. **Run the Script:**
   - Execute the Python script:

     ```bash
     python loan_approval_prediction.py
     ```

## Data Collection

- The script uses a Kaggle dataset on loan prediction (`loan_prediction.csv`).

## Exploratory Data Analysis (EDA)

- Visualizations include pie charts and bar plots to explore loan approval status, gender distribution, marital status distribution, education distribution, self-employment distribution, applicant income distribution, and more.

## Data Cleaning

- Missing values in categorical columns are filled with mode values.
- Missing values in numerical columns are filled with median values.
- Outliers in the "ApplicantIncome" and "CoapplicantIncome" columns are removed.

## Machine Learning Model

- Categorical columns are converted to numerical using one-hot encoding.
- The dataset is split into training and testing sets.
- Numerical columns are scaled using StandardScaler.
- A Support Vector Machine (SVM) model is trained for loan approval prediction.

## Usage

Feel free to customize the script, adjust model parameters, or experiment with different machine learning algorithms.

## License

This project is licensed under the [MIT License](LICENSE), allowing for broad use, modification, and distribution.

