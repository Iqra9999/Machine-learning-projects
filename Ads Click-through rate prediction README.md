# Click-Through Rate Prediction Analysis

## Overview

This Python script analyzes click-through rates (CTR) using the Click-through rate prediction dataset from Kaggle. It explores factors such as time spent on the site, internet usage, age, and income to predict ad click-through rates. The script includes visualizations and a predictive model using the Random Forest classification algorithm.

## Prerequisites

1. **Clone Repository:**
   - Clone this repository to your local machine:

     ```bash
     git clone https://github.com/your-username/click-through-rate-prediction.git
     ```

2. **Install Dependencies:**
   - Ensure you have the required libraries installed:

     ```bash
     pip install pandas plotly numpy scikit-learn
     ```

3. **Run the Script:**
   - Execute the Python script:

     ```bash
     python click_through_rate_analysis.py
     ```

## Dataset

The script uses the Click-through rate prediction dataset from Kaggle (`ad_10000records.csv`). Make sure the dataset is in the project directory.

## Visualizations

- **Time Spent on Site vs. Click Through Rate:** Analyzes CTR based on the time spent by users on the website.
- **Internet Usage vs. Click Through Rate:** Examines CTR based on the daily internet usage of the user.
- **Age vs. Click Through Rate:** Studies CTR based on the age of the user.
- **Income vs. Click Through Rate:** Explores CTR based on the income of the user.

## Click-Through Rate Calculation

- Calculates the overall ads click-through rate.

## Prediction Model

- Uses Random Forest classification to predict ads click-through rates.
- Includes testing the model with user input for predictions.

## Usage

Feel free to customize the script to analyze different datasets or adapt it for your specific prediction needs.

## License

This project is licensed under the [MIT License](LICENSE), allowing for broad use, modification, and distribution.

