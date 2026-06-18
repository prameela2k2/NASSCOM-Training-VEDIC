# Day 3 Notes - Dimensionality Reduction, Class Imbalance, Reproducibility & Optimization

---

# Dimensionality Reduction

Dimensionality Reduction is the process of reducing the number of input features while preserving as much useful information as possible.

Benefits:

* Faster model training
* Reduced storage requirements
* Less overfitting
* Improved visualization
* Reduced noise

---

# Two Ways to Reduce Dimensionality

## 1. Feature Selection

Select a subset of the original features.

Example:

Original Features:

```text
Age
Salary
Experience
City
Department
```

Selected Features:

```text
Age
Salary
Experience
```

Characteristics:

* Original features are retained.
* Easier to interpret.
* Simpler models.

---

## 2. Feature Extraction

Create new features by combining existing features.

Example:

```text
Original Features
      ↓
Transformation
      ↓
New Features
```

Characteristics:

* Original features are transformed.
* May be harder to interpret.
* Often achieves greater dimensionality reduction.

Examples:

* PCA (Principal Component Analysis)
* SVD (Singular Value Decomposition)
* Autoencoders

---

# Feature Selection Methods

Feature selection techniques identify the most important features.

---

## Filter Methods

Features are selected based on statistical tests.

Advantages:

* Fast
* Independent of machine learning models

Examples:

* Chi-Square Test
* ANOVA
* Correlation Analysis

---

### Chi-Square Test

Used for:

```text
Categorical Features
```

Measures the relationship between:

* Feature
* Target Variable

Applications:

* Classification problems
* Categorical datasets

Example:

```text
Gender → Purchased Product
```

Higher Chi-Square value indicates stronger association.

---

### ANOVA (Analysis of Variance)

Used for:

```text
Continuous Features
```

Determines whether group means differ significantly.

Applications:

* Classification problems
* Continuous numerical features

Example:

```text
Salary → Job Category
```

Higher ANOVA score indicates stronger predictive power.

---

# Wrapper Methods

Feature selection is performed by repeatedly training models.

Process:

```text
Select Features
      ↓
Train Model
      ↓
Evaluate Performance
      ↓
Repeat
```

Examples:

* Forward Selection
* Backward Elimination
* Recursive Feature Elimination (RFE)

Advantages:

* Often produces high-quality feature subsets.

Disadvantages:

* Computationally expensive.

---

# Embedded Methods

Feature selection occurs during model training.

Examples:

* Lasso Regression (L1 Regularization)
* Decision Trees
* Random Forests
* Gradient Boosting

Advantages:

* Faster than wrapper methods.
* Considers model performance.
* Generally preferred in practice.

---

## Why Embedded Methods Are Often Preferred

They provide:

* Better efficiency
* Built-in feature selection
* Good predictive performance

Common recommendation:

```text
Filter < Wrapper < Embedded
```

(Trade-off between speed and performance)

---

# Principal Component Analysis (PCA)

PCA is one of the most popular Feature Extraction techniques.

Purpose:

* Reduce dimensionality
* Preserve maximum variance
* Remove redundancy

---

## How PCA Works

Step 1:

Standardize the data.

↓

Step 2:

Calculate covariance matrix.

↓

Step 3:

Find eigenvalues and eigenvectors.

↓

Step 4:

Select principal components.

↓

Step 5:

Project data onto new feature space.

---

## Principal Components

Principal Components are new features created from combinations of original features.

Characteristics:

* Orthogonal to each other
* Capture maximum variance
* Reduce redundancy

---

## Applications of PCA

* Dimensionality Reduction
* Data Compression
* Noise Reduction
* Visualization of High-Dimensional Data
* Machine Learning Preprocessing

---

## Limitation of PCA

* Reduced interpretability
* Assumes linear relationships
* Sensitive to scaling

---

# Handling Class Imbalance

Class imbalance occurs when one class significantly outnumbers another.

Example:

```text
Pass    = 950
Fail    = 50
```

---

## Why Is It a Problem?

A model may predict only the majority class and still achieve high accuracy.

Example:

```text
Predict Everything as Pass

Accuracy = 95%
```

But the model completely fails to identify failures.

---

# Methods to Handle Class Imbalance

## Undersampling

Reduce majority-class samples.

Example:

```text
950 Pass
50 Fail

↓

100 Pass
50 Fail
```

Advantages:

* Faster training

Disadvantages:

* Information loss

---

## Oversampling

Increase minority-class samples.

Example:

```text
950 Pass
50 Fail

↓

950 Pass
950 Fail
```

Advantages:

* Retains all original data

Disadvantages:

* May increase overfitting

---

## SMOTE

Synthetic Minority Oversampling Technique.

Creates synthetic minority samples rather than duplicating existing ones.

Advantages:

* Better generalization
* Reduces overfitting

---

## Use Appropriate Evaluation Metrics

Instead of Accuracy, use:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

---

# Reproducibility

Reproducibility means obtaining the same results when experiments are repeated.

Importance:

* Trustworthy experiments
* Easier debugging
* Consistent research outcomes

---

# Best Practices for Reproducibility

## Fix the Random Seed

Example:

```python
import random
import numpy as np

random.seed(42)
np.random.seed(42)
```

Benefits:

* Consistent results
* Repeatable experiments

---

## Version the Data

Maintain dataset versions.

Example:

```text
customer_data_v1.csv
customer_data_v2.csv
```

Benefits:

* Track changes
* Reproduce historical results

---

## Version the Code

Use version control systems.

Examples:

* Git
* GitHub
* GitLab

Benefits:

* Track modifications
* Collaborate effectively

---

## Log Decisions and Choices

Document:

* Data cleaning steps
* Feature selection methods
* Hyperparameters
* Model choices

Example:

```text
Random Forest
n_estimators = 100
max_depth = 10
seed = 42
```

Benefits:

* Easier debugging
* Better experiment tracking

---

# Optimization

Optimization is the process of finding the best model parameters that minimize error and improve performance.

Goal:

```text
Minimize Loss Function
```

---

## Loss Function

Measures prediction error.

Examples:

* Mean Squared Error (MSE)
* Cross Entropy Loss
* Log Loss

---

## Gradient Descent

Most common optimization algorithm.

Process:

```text
Initialize Parameters
        ↓
Compute Loss
        ↓
Calculate Gradient
        ↓
Update Parameters
        ↓
Repeat Until Convergence
```

Formula:

[
w_{new}
=======

## w_{old}

\eta \frac{\partial L}{\partial w}
]

Where:

* ( \eta ) = Learning Rate
* ( L ) = Loss Function

---

## Types of Gradient Descent

### Batch Gradient Descent

Uses the entire dataset.

Advantages:

* Stable updates

Disadvantages:

* Slow on large datasets

---

### Stochastic Gradient Descent (SGD)

Uses one sample at a time.

Advantages:

* Faster updates

Disadvantages:

* Noisy learning

---

### Mini-Batch Gradient Descent

Uses small batches of data.

Advantages:

* Faster than Batch GD
* More stable than SGD

Most commonly used in Deep Learning.

---

# Key Takeaways

* Dimensionality reduction simplifies datasets.
* Feature Selection keeps original features.
* Feature Extraction creates new features.
* Chi-Square is used for categorical data.
* ANOVA is used for continuous features.
* Embedded methods are often preferred over wrapper methods.
* PCA is a powerful dimensionality reduction technique.
* Class imbalance can make accuracy misleading.
* Use precision, recall, and F1-score for imbalanced datasets.
* Reproducibility requires fixed seeds, versioning, and documentation.
* Optimization aims to minimize loss using algorithms like Gradient Descent.

---
