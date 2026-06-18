# Day 6 Notes - Optimization, Regularization, Supervised Learning & Regression Metrics

---

# Optimization

Optimization is the process of finding model parameters that minimize error and improve performance.

Goal:

```text
Minimize Loss Function
```

---

# Bias-Variance Tradeoff

One of the most important concepts in Machine Learning.

The objective is to balance:

* Bias
* Variance

to achieve good generalization.

---

## Bias

Bias measures how much a model oversimplifies the data.

Characteristics:

* Model is too simple
* Underfits the data
* High training error
* High testing error

Example:

```text
Using a straight line
for highly nonlinear data
```

---

## Variance

Variance measures how sensitive a model is to training data.

Characteristics:

* Model is too complex
* Overfits the data
* Very low training error
* High testing error

Example:

```text
Memorizing training data
instead of learning patterns
```

---

## Bias-Variance Tradeoff

Goal:

```text
Find the balance between
underfitting and overfitting
```

---

### Underfitting

```text
High Bias
Low Variance
```

Model is too simple.

---

### Overfitting

```text
Low Bias
High Variance
```

Model learns noise.

---

### Ideal Model

```text
Low Bias
Low Variance
```

Generalizes well to unseen data.

---

# Global Minima and Local Minima

Optimization algorithms try to find the minimum point of a loss function.

---

## Global Minimum

Lowest possible value of the loss function.

```text
Best Possible Solution
```

---

## Local Minimum

A point where loss is lower than nearby points but not the lowest overall.

```text
Good Solution
But Not Best Solution
```

---

# Momentum

Momentum helps Gradient Descent move faster and avoid oscillations.

Idea:

```text
Current Update
+
Previous Direction
```

Benefits:

* Faster convergence
* Reduced oscillation
* Better navigation through local minima

---

# Adaptive Learning Rates

Instead of using a fixed learning rate, adaptive methods adjust learning rates automatically.

---

## AdaGrad

Adaptive Gradient Algorithm.

Characteristics:

* Larger updates for infrequent features
* Smaller updates for frequent features

Advantages:

* Useful for sparse data

Disadvantages:

* Learning rate can become too small over time

---

## RMSProp

Root Mean Square Propagation.

Improvement over AdaGrad.

Characteristics:

* Prevents learning rate from shrinking too much
* Faster convergence

Advantages:

* Works well for deep learning

---

## Adam Optimizer

Adaptive Moment Estimation.

Combines:

```text
Momentum
+
Adaptive Learning Rates
```

Advantages:

* Fast convergence
* Stable training
* Handles noisy gradients
* Most widely used optimizer

---

## Why Adam Is Popular

Adam combines the strengths of:

* Momentum
* RMSProp

Result:

```text
Fast
Stable
Efficient
```

Note:

While Adam is one of the most popular optimizers, newer optimizers continue to be researched and developed.

---

# Regularization

Regularization helps reduce overfitting by penalizing large model weights.

Goal:

```text
Improve Generalization
```

---

# Ridge Regression (L2 Regularization)

Adds a penalty based on squared coefficients.

Formula:

[
Loss + \lambda \sum w^2
]

Characteristics:

* Shrinks coefficients
* Keeps all features
* Reduces model complexity

Applications:

* Multicollinearity problems
* High-dimensional datasets

---

# Lasso Regression (L1 Regularization)

Adds a penalty based on absolute coefficient values.

Formula:

[
Loss + \lambda \sum |w|
]

Characteristics:

* Some coefficients become exactly zero
* Performs automatic feature selection

Applications:

* Feature selection
* Sparse models

---

# Lasso vs Ridge

| Lasso (L1)              | Ridge (L2)                     |
| ----------------------- | ------------------------------ |
| Feature Selection       | Feature Shrinkage              |
| Can eliminate features  | Keeps all features             |
| Sparse models           | Stable models                  |
| Produces simpler models | Handles multicollinearity well |

---

# Supervised Learning

Supervised Learning learns from labeled data.

Training Data:

```text
Input (X) → Output (Y)
```

The model learns the relationship between inputs and outputs.

---

## Types of Supervised Learning

### Classification

Predicts categories.

Examples:

* Spam / Not Spam
* Pass / Fail
* Disease / No Disease

Output:

```text
Discrete Values
```

---

### Regression

Predicts numerical values.

Examples:

* House Prices
* Sales Forecasting
* Temperature Prediction

Output:

```text
Continuous Values
```

---

# ROC Curve

ROC = Receiver Operating Characteristic Curve

Used for classification model evaluation.

Plots:

```text
True Positive Rate (TPR)
vs
False Positive Rate (FPR)
```

---

# ROC-AUC

AUC = Area Under the Curve

Measures overall classification performance.

Range:

```text
0 to 1
```

Interpretation:

| AUC  | Meaning            |
| ---- | ------------------ |
| 1.0  | Perfect Classifier |
| 0.9+ | Excellent          |
| 0.8+ | Good               |
| 0.5  | Random Guessing    |

---

# Area Under Curve (AUC)

Represents the model's ability to distinguish between classes.

Higher AUC:

```text
Better Classification Performance
```

---

# Regression Metrics

Used to evaluate regression models.

---

# MAE (Mean Absolute Error)

Measures average absolute prediction error.

Formula:

[
MAE =
\frac{1}{n}
\sum |y-\hat{y}|
]

Characteristics:

* Easy to interpret
* Less sensitive to outliers

---

# MSE (Mean Squared Error)

Measures average squared prediction error.

Formula:

[
MSE =
\frac{1}{n}
\sum (y-\hat{y})^2
]

Characteristics:

* Penalizes large errors heavily
* Sensitive to outliers

---

# RMSE (Root Mean Squared Error)

Square root of MSE.

Formula:

[
RMSE=
\sqrt{MSE}
]

Characteristics:

* Same units as target variable
* Easier interpretation

---

# R² Score (Coefficient of Determination)

Measures how much variance is explained by the model.

Formula:

[
R^2=
1-\frac{SS_{res}}{SS_{tot}}
]

Range:

```text
≤ 1
```

Interpretation:

| R² Value | Meaning                    |
| -------- | -------------------------- |
| 1.0      | Perfect Fit                |
| 0.0      | No Improvement Over Mean   |
| < 0      | Worse Than Mean Prediction |

---

# Linear Regression

One of the simplest supervised learning algorithms.

Purpose:

```text
Predict Continuous Values
```

---

## Equation of a Line

Linear Regression learns:

[
y = mx + b
]

Where:

* y = Predicted Value
* x = Input Feature
* m = Slope (Weight)
* b = Intercept (Bias)

---

## Goal of Linear Regression

Find the best-fitting line that minimizes prediction error.

---

## Applications

* House Price Prediction
* Sales Forecasting
* Demand Prediction
* Trend Analysis

---

# Learning Resources

## Scikit-Learn

A popular Python machine learning library.

Topics Covered:

* Classification
* Regression
* Clustering
* Preprocessing
* Model Evaluation

Official Documentation:

https://scikit-learn.org/stable/

---

## Stanford Machine Learning

Widely recognized machine learning learning material covering:

* Supervised Learning
* Unsupervised Learning
* Optimization
* Neural Networks
* Deep Learning Foundations

---

# Key Takeaways

* Bias causes underfitting.
* Variance causes overfitting.
* The goal is to balance bias and variance.
* Global minima is the best possible solution.
* Momentum accelerates gradient descent.
* AdaGrad and RMSProp use adaptive learning rates.
* Adam combines Momentum and Adaptive Learning Rates.
* Regularization reduces overfitting.
* Lasso performs feature selection.
* Ridge shrinks coefficients while keeping all features.
* Supervised learning uses labeled data.
* Classification predicts categories.
* Regression predicts numerical values.
* ROC-AUC evaluates classification models.
* MAE, MSE, RMSE, and R² evaluate regression models.
* Linear Regression is the foundation of many predictive models.

---
