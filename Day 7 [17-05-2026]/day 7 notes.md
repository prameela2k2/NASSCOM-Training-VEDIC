# Day 7 Notes - Elastic Net, Outlier Handling, Q-Q Plots, Transformations & Classification

---

# Elastic Net Regression

Elastic Net combines both:

* L1 Regularization (Lasso)
* L2 Regularization (Ridge)

Formula:

[
Loss +
\lambda_1 \sum |w|
+
\lambda_2 \sum w^2
]

---

## Why Elastic Net?

Lasso may remove too many features.

Ridge keeps all features.

Elastic Net provides a balance between both.

Benefits:

* Feature Selection
* Coefficient Shrinkage
* Handles Multicollinearity
* Better stability than Lasso alone

---

# Outlier Handling

Outliers are unusually large or small values.

Example:

```text
10, 12, 11, 13, 15, 500
```

Here, 500 is an outlier.

---

## Dropping Outliers

Remove records containing extreme values.

Advantages:

* Simple approach
* Removes noise

Disadvantages:

* Loss of information
* Smaller dataset

---

## Capping Outliers

Replace extreme values with threshold values.

Example:

```text
Values above 95th percentile
→ replaced by 95th percentile value
```

Advantages:

* Preserves dataset size
* Reduces outlier influence

---

# Q-Q Plot (Quantile-Quantile Plot)

A graphical method used to compare a dataset against a theoretical distribution.

Most commonly:

```text
Sample Distribution
vs
Normal Distribution
```

---

## Purpose

Check whether data follows a normal distribution.

---

## Interpretation

### Points Follow a Straight Line

```text
Data is approximately normal
```

---

### Points Deviate from the Line

```text
Data is not normally distributed
```

Possible reasons:

* Skewness
* Heavy tails
* Outliers

---

# Data Transformations

Transformations help make data more suitable for regression models.

Goals:

* Reduce skewness
* Stabilize variance
* Improve normality
* Improve model performance

---

# Box-Cox Transformation

A power transformation technique.

Characteristics:

* Works only with positive values
* Helps normalize data
* Reduces skewness

Requirements:

```text
Values > 0
```

Applications:

* Regression problems
* Positively skewed data

---

# Yeo-Johnson Transformation

An extension of Box-Cox.

Characteristics:

* Handles positive values
* Handles zero values
* Handles negative values

Advantages:

* More flexible than Box-Cox
* No positivity requirement

---

# Important Note

In many regression workflows, transformations such as:

* Box-Cox
* Yeo-Johnson

can be applied to the target variable (Y) to make it more normally distributed and easier to model.

They may also be applied to features (X) when appropriate, but transforming the target variable is a very common use case in regression.

---

# Classification

Classification predicts categories or classes.

Output:

```text
Discrete Labels
```

Examples:

* Spam / Not Spam
* Pass / Fail
* Disease / No Disease

---

# Types of Classification

## Binary Classification

Two classes.

Examples:

```text
Yes / No
0 / 1
True / False
```

---

## Multiclass Classification

More than two classes.

Examples:

```text
Cat
Dog
Horse
```

Only one class is selected.

---

## Multi-Label Classification

A single observation can belong to multiple classes.

Example:

```text
Movie Genres

Action
Comedy
Drama
```

A movie may belong to multiple genres simultaneously.

---

# Sigmoid Function

Used in Logistic Regression.

Converts any real number into a probability between 0 and 1.

Formula:

[
\sigma(z)
=========

\frac{1}
{1+e^{-z}}
]

---

## Characteristics

Output Range:

```text
0 to 1
```

Shape:

```text
S-Curve
```

Applications:

* Logistic Regression
* Binary Classification
* Neural Networks

---

## Interpretation

Example:

```text
Output = 0.90
```

Meaning:

```text
90% probability
of belonging to Class 1
```

---

# Elbow Method

Used to determine the optimal number of clusters in K-Means Clustering.

Process:

1. Run K-Means for different values of K.
2. Calculate Within-Cluster Sum of Squares (WCSS).
3. Plot K vs WCSS.
4. Identify the "elbow" point.

---

## Why Called Elbow?

The graph often looks like:

```text
\
 \
  \
   \__
      \__
```

The bend resembles an elbow.

---

## Optimal K

The elbow point is chosen as the best number of clusters.

---

# Why Are We Unable to Apply Gradient Descent Directly on Classification Problems?

This is a common interview question.

---

## Problem with Linear Regression for Classification

Suppose:

```text
Class 0
Class 1
```

If we use Linear Regression:

```text
Output can be:

-10
2.5
100
```

These values are not valid probabilities.

---

## Non-Convex Cost Function

If we directly use classification labels with a linear regression-style classification objective, the resulting optimization can lead to a non-convex error surface with multiple local minima.

Gradient Descent may struggle to find a good solution.

---

## Solution: Logistic Regression

Instead of predicting raw values:

```text
(-∞ , +∞)
```

Logistic Regression applies the Sigmoid Function.

Result:

```text
0 ≤ Probability ≤ 1
```

---

## Why Gradient Descent Works in Logistic Regression

Logistic Regression uses:

```text
Cross Entropy Loss
(Log Loss)
```

Properties:

* Differentiable
* Convex for logistic regression
* Suitable for Gradient Descent

Therefore:

```text
Gradient Descent
+
Sigmoid Function
+
Cross Entropy Loss

=
Efficient Classification Learning
```

---

# Key Takeaways

* Elastic Net combines Lasso and Ridge.
* Dropping and capping are common outlier treatments.
* Q-Q plots help assess normality.
* Box-Cox works only with positive values.
* Yeo-Johnson handles positive, zero, and negative values.
* Classification predicts categories.
* Binary classification has two classes.
* Multiclass classification predicts one among many classes.
* Multi-label classification predicts multiple classes simultaneously.
* Sigmoid converts outputs into probabilities.
* Elbow Method helps determine the optimal number of clusters.
* Logistic Regression uses Sigmoid and Cross-Entropy Loss for classification.
* Gradient Descent works effectively because the optimization objective is differentiable and well-behaved.

---
