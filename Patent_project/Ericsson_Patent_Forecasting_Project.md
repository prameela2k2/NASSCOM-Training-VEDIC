# Forecasting Ericsson Patent Activity Using Linear Regression, Ridge Regression and Lasso Regression

## 1. Project Overview

### Project Title

**Forecasting Ericsson Patent Activity Using Linear Regression, Ridge Regression and Lasso Regression**

### Objective

The primary objective of this project is to forecast Ericsson's future patent activity using historical patent records and technology-related indicators. Machine Learning regression techniques are employed to predict the number of patents expected in the next quarter.

### Target Variable

```python
target_patent_count_next_quarter
```

The target variable represents the expected patent count in the subsequent quarter based on historical trends and technological innovation indicators.

---

# 2. Dataset Description

### Dataset Name

Ericsson Innovation Timeline: Patent Evolution

### Source

```python
python -m kaggle datasets download -d adamvakar/ericsson-innovation-timeline-patent-evolution
```

### Dataset Size

* Total Records: 30,118
* Total Features: 55

### Key Features

| Feature                 | Description                                         |
| ----------------------- | --------------------------------------------------- |
| year                    | Patent filing year                                  |
| quarter                 | Patent filing quarter                               |
| keyword_score           | Technology keyword relevance score                  |
| patent_count_lag1       | Previous quarter patent count                       |
| patent_count_lag2       | Patent count two quarters earlier                   |
| patent_count_lag4       | Patent count four quarters earlier                  |
| patent_count_roll4_mean | Rolling average of patent counts over four quarters |
| patent_count_roll8_mean | Rolling average over eight quarters                 |
| kw_ai_ml                | AI/ML-related patent indicator                      |
| kw_5g                   | 5G-related patent indicator                         |
| kw_network              | Network technology indicator                        |
| tech_era                | Technology era classification                       |
| patent_type             | Patent category                                     |

---

# 3. CRISP-DM Methodology

The project follows the CRISP-DM (Cross Industry Standard Process for Data Mining) lifecycle:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment Readiness

---

# 4. Data Understanding

Initial data exploration was performed using:

```python
df.head()
df.info()
df.describe()
```

### Dataset Information

* Rows: 30,118
* Columns: 55
* Memory Usage: 12.6 MB

Data types include:

* Integer Features
* Floating Point Features
* Date-Time Features
* Categorical Features

---

# 5. Data Cleaning

## Missing Value Analysis

The following column contained missing values:

```python
target_patent_count_next_quarter
```

Missing Values:

```text
408
```

These missing values occurred because the final quarter of the dataset does not have future patent count information available.

### Handling Missing Values

Rows containing missing target values were removed:

```python
df = df.dropna(
    subset=['target_patent_count_next_quarter']
)
```

Final Dataset Size:

```text
29,710 records
```

---

## Duplicate Analysis

Duplicate records were checked using:

```python
df.duplicated().sum()
```

Any duplicate observations were removed if present.

---

# 6. Exploratory Data Analysis (EDA)

## Patent Activity by Year

Patent counts were aggregated annually to analyze long-term innovation trends.

### Insights

* Significant growth observed in recent years.
* Patent filing activity increased with emerging technologies.
* Innovation acceleration visible during modern telecommunications eras.

---

## Patent Activity by Quarter

Quarter-wise patent distributions were analyzed to identify seasonal patterns.

### Insights

* Patent activity remained relatively consistent across quarters.
* No major seasonal fluctuations were observed.

---

## Keyword Score Distribution

The distribution of keyword scores was visualized using histograms.

### Insights

* Most patents contain moderate technology relevance scores.
* Distribution exhibited slight skewness toward higher innovation-related keywords.

---

## AI/ML Patent Trend Analysis

AI and Machine Learning patent indicators were analyzed over time.

### Insights

* Continuous growth observed in AI/ML-related patents.
* Indicates increasing investment in intelligent communication systems.

---

## 5G Patent Trend Analysis

5G-related patents were analyzed across years.

### Insights

* Rapid growth observed after the emergence of 5G technologies.
* Demonstrates Ericsson's strong focus on next-generation communication infrastructure.

---

# 7. Correlation Analysis

A correlation matrix and heatmap were generated to identify relationships among variables.

### Top Features Correlated with Future Patent Activity

| Feature                 | Correlation |
| ----------------------- | ----------- |
| year_patent_count       | 0.954       |
| patent_count            | 0.929       |
| patent_count_roll4_mean | 0.923       |
| kw_network_count        | 0.907       |
| patent_count_lag1       | 0.900       |
| patent_count_roll8_mean | 0.899       |
| patent_count_lag2       | 0.881       |
| patent_count_lag4       | 0.854       |

### Interpretation

Historical patent activity and rolling trend features strongly influence future patent counts.

---

# 8. Feature Engineering

Selected features for modeling:

```python
features = [
    'year',
    'quarter',
    'title_len_words',
    'keyword_score',
    'kw_ai_ml',
    'kw_5g',
    'kw_cloud_edge',
    'kw_security',
    'kw_iot',
    'kw_network',
    'patent_count_lag1',
    'patent_count_lag2',
    'patent_count_lag4',
    'patent_count_roll4_mean',
    'patent_count_roll8_mean',
    'patent_count_qoq',
    'patent_count_yoy'
]
```

Target:

```python
target = 'target_patent_count_next_quarter'
```

---

# 9. Categorical Feature Encoding

The following categorical features were encoded using One-Hot Encoding:

```python
patent_type
tech_era
```

Implementation:

```python
pd.get_dummies(
    df,
    columns=['patent_type','tech_era'],
    drop_first=True
)
```

---

# 10. Data Preprocessing

## Train-Test Split

```python
from sklearn.model_selection import train_test_split
```

Configuration:

```python
test_size = 0.20
random_state = 42
```

---

## Feature Scaling

Standardization was performed using:

```python
from sklearn.preprocessing import StandardScaler
```

Purpose:

* Normalize feature ranges.
* Improve Ridge and Lasso Regression performance.
* Prevent scale dominance.

---

# 11. Machine Learning Models

## Linear Regression

Linear Regression establishes a linear relationship between predictor variables and future patent counts.

### Advantages

* Easy to interpret
* Fast training
* Strong baseline model

---

## Ridge Regression

Ridge Regression applies L2 Regularization.

### Advantages

* Reduces coefficient variance
* Handles multicollinearity
* Prevents overfitting

---

## Lasso Regression

Lasso Regression applies L1 Regularization.

### Advantages

* Performs automatic feature selection
* Reduces model complexity
* Produces sparse models

---

# 12. Model Evaluation Metrics

The following evaluation metrics were used:

## Mean Squared Error (MSE)

Measures average squared prediction error.

```python
from sklearn.metrics import mean_squared_error
```

## R² Score

Measures the proportion of variance explained by the model.

```python
from sklearn.metrics import r2_score
```

---

# 13. Experimental Results

| Model             | MSE     | R² Score |
| ----------------- | ------- | -------- |
| Linear Regression | 1586.60 | 0.8851   |
| Ridge Regression  | 1586.63 | 0.8851   |
| Lasso Regression  | 1586.68 | 0.8851   |

---

# 14. Results Discussion

### Linear Regression

The model achieved excellent predictive performance, explaining approximately 88.5% of future patent activity variation.

### Ridge Regression

Ridge Regression produced nearly identical results, indicating limited multicollinearity issues within the selected feature set.

### Lasso Regression

Lasso Regression also achieved similar performance, suggesting that most selected features contribute meaningful predictive information.

---

# 15. Conclusion

This project successfully developed a patent forecasting framework for Ericsson using Linear Regression, Ridge Regression, and Lasso Regression.

Comprehensive data preprocessing, exploratory analysis, feature engineering, and model evaluation were performed on a large-scale patent dataset. Historical patent activity variables, technology indicators, and trend-based features proved highly effective in predicting future patent counts.

The best-performing model achieved an R² score of approximately 0.885, demonstrating strong predictive capability and validating the effectiveness of regression-based forecasting approaches for innovation analytics.

---

# 16. Future Scope

Future enhancements may include:

1. Random Forest Regression
2. Gradient Boosting Regression
3. XGBoost Regression
4. Support Vector Regression
5. ARIMA Forecasting
6. Prophet Time-Series Models
7. LSTM Neural Networks
8. Patent Text Mining using NLP
9. Integration of External R&D Data
10. Real-Time Patent Forecasting Dashboard

These improvements can further enhance forecasting accuracy and provide deeper insights into technological innovation trends.

---
# 17. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* SciPy
* Jupyter Notebook

---

# 18. References

1. Ericsson Patent Evolution Dataset (Kaggle)
2. Scikit-Learn Documentation
3. Pandas Documentation
4. CRISP-DM Methodology Guide
5. Machine Learning Regression Literature
