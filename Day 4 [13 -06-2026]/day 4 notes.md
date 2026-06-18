# Day 4 Notes - Data Cleaning, Preprocessing, Feature Scaling & EDA

---

# Data Cleaning and Preprocessing

Data preprocessing transforms raw data into a clean and usable format for machine learning.

Goals:

* Improve data quality
* Handle missing values
* Handle outliers
* Standardize data
* Prepare data for modeling

---

# Feature Engineering

Feature Engineering is the process of creating, transforming, or selecting features that help machine learning models perform better.

Examples:

* Extracting year from a date
* Creating age from date of birth
* Combining multiple features
* Encoding categorical variables

Benefits:

* Improves model accuracy
* Reduces noise
* Enhances interpretability

---

# Feature Scaling

Feature scaling ensures all features are on a similar scale.

Why is it needed?

Consider:

```text
Age = 25
Salary = 500000
```

The salary feature dominates because of its larger magnitude.

Scaling prevents this issue.

---

## Standardization (Z-Score Normalization)

Formula:

[
z=\frac{x-\mu}{\sigma}
]

Where:

* μ = Mean
* σ = Standard Deviation

Characteristics:

* Mean becomes 0
* Standard deviation becomes 1

Used in:

* Logistic Regression
* SVM
* PCA
* Neural Networks

---

## Min-Max Scaling

Formula:

[
x'=\frac{x-min}{max-min}
]

Range:

```text
0 to 1
```

Used when bounded values are preferred.

---

## Robust Scaling

Uses:

* Median
* Interquartile Range (IQR)

Less sensitive to outliers.

---

# Pipelines

A Pipeline automates preprocessing and model-building steps.

Example Workflow:

```text
Data
 ↓
Missing Value Handling
 ↓
Scaling
 ↓
Feature Engineering
 ↓
Model Training
```

Benefits:

* Reduces manual work
* Prevents data leakage
* Ensures reproducibility

Example:

```python
from sklearn.pipeline import Pipeline
```

---

# Exploratory Data Analysis (EDA)

EDA helps understand data before model building.

Goals:

* Understand distributions
* Detect outliers
* Find relationships
* Identify patterns

---

# EDA Workflow

## Step 1: Understand Data

Check:

```python
df.head()
df.info()
df.describe()
```

---

## Step 2: Data Cleaning

Handle:

* Missing values
* Duplicates
* Outliers

---

## Step 3: Univariate Analysis

Analyze one variable at a time.

Tools:

* Histogram
* Box Plot
* Density Plot

---

## Step 4: Bivariate Analysis

Analyze relationships between two variables.

Examples:

* Scatter Plot
* Correlation Analysis

---

## Step 5: Multivariate Analysis

Analyze relationships among multiple variables.

Examples:

* Heatmaps
* Pair Plots

---

# Univariate Analysis

Study of a single variable.

Questions answered:

* Distribution?
* Center?
* Spread?
* Outliers?

---

## Histogram

Shows frequency distribution of data.

Example:

```python
plt.hist(data)
```

Uses:

* Distribution shape
* Skewness
* Spread

---

# Kurtosis

Measures the heaviness of tails in a distribution.

---

## High Kurtosis (Leptokurtic)

Characteristics:

* Heavy tails
* More outliers

---

## Low Kurtosis (Platykurtic)

Characteristics:

* Light tails
* Fewer outliers

---

## Normal Distribution

Kurtosis ≈ 3

(Many libraries report Excess Kurtosis, where Normal Distribution = 0)

---

# Box Plots by Category

Used to compare distributions across categories.

Example:

```python
sns.boxplot(
    x="Department",
    y="Salary",
    data=df
)
```

Benefits:

* Detect outliers
* Compare groups
* Visualize spread

---

# Correlation Analysis

Measures relationship between numerical variables.

Range:

[
-1 \le r \le 1
]

Values:

| Correlation | Meaning          |
| ----------- | ---------------- |
| +1          | Perfect Positive |
| 0           | No Relationship  |
| -1          | Perfect Negative |

---

# Correlation Heatmap

Visual representation of correlation matrix.

Example:

```python
sns.heatmap(df.corr(), annot=True)
```

Benefits:

* Quickly identify strong relationships
* Detect multicollinearity
* Feature selection

---

# Why Do We Write %matplotlib inline?

```python
%matplotlib inline
```

Used mainly in:

* Jupyter Notebook
* Google Colab

Purpose:

Display plots directly inside the notebook output cells instead of opening separate windows.

Note:

Modern versions of Jupyter and Colab often display plots without explicitly writing this command, but it is still commonly used for compatibility and consistency.

---

# Matplotlib

Matplotlib is Python's primary plotting library.

The name comes from:

```text
MATLAB + Plotting
```

It was inspired by MATLAB's plotting capabilities and built using Python's scientific ecosystem, including:

* NumPy
* Python Scientific Libraries

---

# Subplots

Subplots allow multiple plots within a single figure.

Example:

```python
fig, ax = plt.subplots(2,2)
```

Creates:

```text
┌─────┬─────┐
│Plot1│Plot2│
├─────┼─────┤
│Plot3│Plot4│
└─────┴─────┘
```

Benefits:

* Compare multiple charts
* Save space
* Better visualization

---

# Error Bar Plot

Displays uncertainty or variability in data.

Example:

```python
plt.errorbar(
    x,
    y,
    yerr=error
)
```

Components:

* Central value
* Error range

Applications:

* Scientific experiments
* Statistical analysis
* Model evaluation

Example:

```text
Mean Accuracy = 90%
Error = ±2%
```

Visualizes confidence around measurements.

---

# Key Takeaways

* Preprocessing prepares data for machine learning.
* Feature Engineering creates useful predictors.
* Scaling prevents features with large values from dominating.
* Pipelines automate preprocessing and training.
* EDA helps understand data before modeling.
* Histograms reveal distributions.
* Kurtosis measures tail heaviness.
* Box plots help identify outliers.
* Correlation measures relationships.
* Heatmaps visualize correlation matrices.
* `%matplotlib inline` displays plots within notebooks.
* Subplots show multiple charts in one figure.
* Error bars represent uncertainty and variability.

---
