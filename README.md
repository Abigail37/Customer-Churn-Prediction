# Customer Churn Prediction Using Machine Learning

A machine-learning project that analyzes telecom customer data and predicts whether a customer is likely to churn.

## Project Overview

Customer churn is an important business problem for subscription-based companies. Identifying customers who are likely to leave can help businesses develop targeted retention strategies.

In this project, I explored a telecom customer dataset, identified patterns associated with churn, prepared the data for machine learning, trained classification models, evaluated their performance, experimented with classification thresholds, and interpreted the final model.

The main objective was not only to predict churn but also to understand **which customer characteristics are associated with higher or lower churn risk**.

## Dataset

The dataset contains information about **7,043 telecom customers**.

The features include information about:

* Customer demographics
* Tenure
* Contract type
* Internet services
* Phone services
* Online security
* Technical support
* Streaming services
* Payment methods
* Monthly charges
* Total charges

The target variable is:

* `0` — Customer stayed
* `1` — Customer churned

## Key Exploratory Findings

The dataset was imbalanced:

* Approximately **73%** of customers stayed.
* Approximately **27%** churned.

Contract type showed one of the strongest relationships with churn:

| Contract Type  | Churn Rate |
| -------------- | ---------: |
| Month-to-month |      42.7% |
| One year       |      11.3% |
| Two year       |       2.8% |

Other notable patterns included higher churn among customers using electronic check payments and a relatively high churn rate among fiber-optic customers.

## Data Preparation

Categorical features were identified using pandas and converted into numerical variables using one-hot encoding.

```python
categorical_cols = df.select_dtypes(include='object').columns

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    dtype=int
)
```

The resulting feature matrix contained:

```text
7,043 rows
30 features
```

The target variable was already encoded as `0` and `1`.

The dataset was split using an 80/20 train-test split:

```text
Training set: 5,634 customers
Testing set: 1,409 customers
```

## Models Tested

Two main approaches were tested:

1. Logistic Regression
2. Random Forest

A balanced Random Forest was also tested to determine whether giving additional weight to the churn class would improve churn detection.

## Model Performance

### Logistic Regression — Threshold 0.50

| Metric          | Result |
| --------------- | -----: |
| Accuracy        | 82.19% |
| Churn Precision |    69% |
| Churn Recall    |    60% |
| Churn F1-score  |    64% |
| ROC-AUC         | 0.8625 |

The confusion matrix was:

```text
[[934, 102],
 [149, 224]]
```

The model correctly identified 224 of the 373 customers who actually churned.

### Random Forest

| Metric       | Result |
| ------------ | -----: |
| Accuracy     | 78.92% |
| Churn Recall |    46% |

### Balanced Random Forest

| Metric          | Result |
| --------------- | -----: |
| Churn Precision |    64% |
| Churn Recall    |    49% |
| Churn F1-score  |    56% |

The Random Forest models performed worse than Logistic Regression for the main objective of identifying churners.

## Threshold Tuning

The default Logistic Regression classification threshold was 0.50.

I experimented with lowering the threshold to **0.30**.

### Threshold 0.30

| Metric          | Result |
| --------------- | -----: |
| Accuracy        |  77.5% |
| Churn Precision |    55% |
| Churn Recall    |    80% |
| Churn F1-score  |    65% |

The change significantly improved churn detection.

At the 0.50 threshold:

* 224 of 373 actual churners were identified.
* 149 churners were missed.

At the 0.30 threshold:

* 299 of 373 actual churners were identified.
* Only 74 churners were missed.

This represents **75 additional churners identified**.

The trade-off was a reduction in precision and overall accuracy because the model became more aggressive in flagging potential churners.

## Model Interpretation

The Logistic Regression coefficients were examined to understand which features were associated with higher or lower predicted churn risk.

### Features associated with higher churn risk

Some of the strongest positive coefficients included:

| Feature                           | Coefficient |
| --------------------------------- | ----------: |
| Fiber optic internet              |        0.91 |
| Paperless billing                 |        0.33 |
| Multiple lines / no phone service |        0.33 |
| Streaming movies                  |        0.33 |
| Electronic check                  |        0.31 |
| Multiple lines                    |        0.27 |
| Streaming TV                      |        0.23 |
| Senior citizen                    |        0.16 |

### Features associated with lower churn risk

| Feature           | Coefficient |
| ----------------- | ----------: |
| Two-year contract |       -1.39 |
| One-year contract |       -0.63 |
| Online security   |       -0.41 |
| Tech support      |       -0.33 |
| Phone service     |       -0.16 |
| Online backup     |       -0.15 |
| Dependents        |       -0.15 |

### Odds Ratios

Two notable coefficients were converted into odds ratios.

For two-year contracts:

```python
np.exp(-1.39)
```

Approximately:

```text
0.25
```

For fiber-optic internet:

```python
np.exp(0.91)
```

Approximately:

```text
2.48
```

These values describe associations in the model and should not be interpreted as proof of causation.

## Business Insights

The analysis suggests several areas that a telecom company could investigate:

### 1. Contract type matters

Month-to-month customers showed dramatically higher churn than customers on one-year or two-year contracts.

The company could investigate incentives that encourage customers to move to longer-term contracts.

### 2. Fiber-optic customers require attention

Fiber-optic service had one of the strongest positive coefficients in the Logistic Regression model.

This does not prove that fiber service causes churn, but it suggests that this customer segment deserves further investigation.

Possible areas to investigate include pricing, service quality, customer expectations, and technical support.

### 3. Payment method may be useful for segmentation

Electronic check customers showed a relatively high churn rate and the corresponding model feature had a positive coefficient.

This could be useful for targeted customer analysis.

### 4. Retention services may be associated with lower churn

Customers with services such as online security and technical support showed negative coefficients.

These services may be worth investigating as part of customer retention strategies.

## Final Recommendation

For general prediction at the default threshold, Logistic Regression performed better than the Random Forest models tested.

However, if the business objective is specifically to **identify as many potential churners as possible**, the Logistic Regression model with a **0.30 classification threshold** is more attractive because it achieved approximately **80% churn recall**.

The final threshold should ultimately depend on the cost of:

* Missing a customer who will churn.
* Contacting a customer who would have stayed.
* Offering discounts or retention incentives.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook

## Machine Learning Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Categorical Encoding
      ↓
Feature / Target Separation
      ↓
Train-Test Split
      ↓
Logistic Regression
      ↓
Prediction
      ↓
Model Evaluation
      ↓
Random Forest Comparison
      ↓
Threshold Tuning
      ↓
ROC-AUC Evaluation
      ↓
Model Interpretation
      ↓
Business Insights
```

## Project Structure

```text
Customer Churn Prediction/
│
├── data/
│   └── customer_churn.csv
│
├── notebooks/
│   └── Customer Churn Analysis & Prediction.ipynb
│
├── images/
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── README.md
```

## Conclusion

This project demonstrates an end-to-end machine-learning workflow for customer churn prediction.

The Logistic Regression model achieved an **82.19% accuracy** and an **ROC-AUC of 0.8625** at the default threshold.

More importantly, threshold tuning showed that the model could increase churn recall from **60% to 80%** by lowering the classification threshold from 0.50 to 0.30.

The project reinforced an important machine-learning principle:

> **The best model is not necessarily the model with the highest accuracy. The right model depends on the business problem, the cost of different types of errors, and the metric that matters most.**

**Author: Abigail Ogunmola**
Dataset from Kaggle