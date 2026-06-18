# Day 2 Notes - Mathematics, Python, Probability & Statistics for AI/ML

---

# 1. Linear Algebra: The Language of Data, Models & Transformations

## Scalars, Vectors, Matrices and Tensors

### Scalar
A single numerical value.

Example:
```python
x = 5
```

### Vector
A collection of numbers arranged in one dimension.

Example:

\[
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
\]

### Matrix
A rectangular arrangement of numbers.

Example:

\[
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\]

### Tensor
A multi-dimensional array.

Examples:

- Scalar → 0D Tensor
- Vector → 1D Tensor
- Matrix → 2D Tensor
- Image Dataset → 3D/4D Tensor

---

## Dot Product

Measures similarity between vectors.

Formula:

\[
A \cdot B = \sum_{i=1}^{n} A_i B_i
\]

Example:

```python
A = [1,2,3]
B = [4,5,6]

Dot Product = 1×4 + 2×5 + 3×6 = 32
```

Applications:

- Similarity measurement
- Neural networks
- Projections

---

## Cross Product

Applicable only for 3D vectors.

Produces a vector perpendicular to both vectors.

\[
A \times B
\]

Applications:

- Physics
- Computer Graphics
- Robotics

---

## Norms

Measure the magnitude (length) of vectors.

### L1 Norm

\[
||x||_1 = \sum |x_i|
\]

### L2 Norm (Euclidean Norm)

\[
||x||_2 = \sqrt{\sum x_i^2}
\]

### Infinity Norm

\[
||x||_\infty = \max(|x_i|)
\]

Applications:

- Regularization
- Distance calculations

---

## Cosine Similarity

Measures angle similarity between vectors.

Formula:

\[
\text{Cosine Similarity}
=
\frac{A \cdot B}
{||A||\,||B||}
\]

Range:

- +1 → Identical direction
- 0 → Orthogonal
- -1 → Opposite direction

Applications:

- Recommendation Systems
- NLP
- Embeddings

---

# Matrix Operations

## Addition

\[
A + B
\]

Both matrices must have same dimensions.

---

## Subtraction

\[
A - B
\]

---

## Matrix Multiplication

\[
AB
\]

Columns of A must equal rows of B.

---

## Transpose

Rows become columns.

\[
A^T
\]

---

## Inverse

\[
A^{-1}
\]

Used to solve systems of equations.

---

# Matrices as Actions (Transformations)

Matrices can transform vectors.

## Scaling

Changes size.

Example:

\[
\begin{bmatrix}
2 & 0 \\
0 & 2
\end{bmatrix}
\]

Doubles object size.

---

## Rotation

Rotates vectors around origin.

Rotation matrix:

\[
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\]

Applications:

- Computer Graphics
- Robotics

---

## Shear

Slants objects.

Example:

\[
\begin{bmatrix}
1 & k \\
0 & 1
\end{bmatrix}
\]

---

## Reflection

Mirrors objects.

Reflection across x-axis:

\[
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
\]

---

# Linear Transformations

A transformation T is linear if:

\[
T(a+b)=T(a)+T(b)
\]

and

\[
T(ca)=cT(a)
\]

Examples:

- Rotation
- Scaling
- Reflection

Not Linear:

- Translation

---

# Eigenvalues and Eigenvectors

For matrix A:

\[
Av = \lambda v
\]

Where:

- v = Eigenvector
- λ = Eigenvalue

Meaning:

Direction remains unchanged after transformation.

Applications:

## PCA

Principal Component Analysis

Used for:

- Dimensionality Reduction
- Feature Extraction

---

## SVD

Singular Value Decomposition

Used in:

- Recommender Systems
- Compression
- NLP

---

## Stability Analysis

Used to determine system behavior over time.

---

# Structure of Matrix

## Gaussian Elimination

Used to solve linear equations.

Steps:

1. Convert matrix to row-echelon form.
2. Perform row operations.
3. Back-substitution.

Applications:

- Solving systems of equations
- Matrix rank calculation

---

# Similarity and Embeddings

## Embeddings

Convert objects into vectors.

Example:

```
King → [0.4,0.8,...]
Queen → [0.5,0.7,...]
```

Similar concepts have nearby vectors.

---

## Vector Embeddings

Represent:

- Words
- Images
- Users
- Products

Applications:

- Search Engines
- Recommendation Systems
- LLMs

---

## Cosine Similarity

Used to compare embeddings.

Higher similarity = More related.

---

## Nearest Neighbours

Find closest vectors.

Applications:

- Search
- Recommendations
- Classification

---

## One-Hot Encoding

Categorical variable representation.

Example:

| Color | Red | Green | Blue |
|---------|---------|---------|---------|
| Red | 1 | 0 | 0 |
| Green | 0 | 1 | 0 |
| Blue | 0 | 0 | 1 |

---

## Bag of Words

Text representation technique.

Example:

Sentence:

```
I love AI
```

Vocabulary:

```
[I, love, AI]
```

Vector:

```
[1,1,1]
```

---

# Why Random Seed = 42?

```python
random.seed(42)
```

A seed ensures reproducible random results.

Reason for 42:

- Popularized by *The Hitchhiker's Guide to the Galaxy*
- Known as "The Answer to Life, the Universe and Everything"
- Widely adopted by programmers as a fun convention

---

# Hands-on Linear Algebra

Topics Practiced:

- Vector Operations
- Matrix Operations
- Dot Product
- Cosine Similarity
- Matrix Multiplication
- Eigenvalues and Eigenvectors

---

# Gradient Descent

Optimization algorithm used to minimize loss.

Formula:

\[
w_{new}
=
w_{old}
-
\eta \frac{\partial L}{\partial w}
\]

Where:

- η = Learning Rate
- L = Loss

---

## Forward Pass

Data flows:

Input → Hidden Layer → Output

Produces prediction.

---

## Backpropagation

Calculates gradients using Chain Rule.

Flow:

Output → Hidden Layer → Input

Updates weights.

---

# Hands-on Calculus

Topics Covered:

- Derivatives
- Partial Derivatives
- Chain Rule
- Gradient
- Backpropagation

---

# Python for AI/ML

## NumPy

Numerical computing library.

### ndarray

Fundamental NumPy data structure.

Example:

```python
import numpy as np

arr = np.array([1,2,3])
```

---

### Create and Shape

```python
arr.shape
arr.reshape()
```

---

### Slicing and Indexing

```python
arr[0]
arr[1:4]
```

---

# Pandas

Data manipulation library.

## Loading Data

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

---

## Indexing

```python
df.loc[]
df.iloc[]
```

---

## Data Cleaning

```python
df.dropna()
df.fillna()
```

---

## Grouping

```python
df.groupby()
```

---

# Matplotlib

Visualization library.

Example:

```python
import matplotlib.pyplot as plt

plt.plot(x,y)
plt.show()
```

---

# Seaborn

Statistical visualization library.

Example:

```python
import seaborn as sns

sns.histplot(data)
```

---

# Hands-on NumPy and Python Part 2

Topics Practiced:

- Arrays
- Matrix Operations
- Data Manipulation
- Visualization
- Aggregations

---

# Probability and Statistics

---

## Types of Data

### Nominal Data

Categories with no order.

Examples:

- Gender
- Color
- Country

---

### Ordinal Data

Categories with meaningful order.

Examples:

- Low
- Medium
- High

---

# Measures of Central Tendency

## Mean

Average value.

\[
\text{Mean}
=
\frac{\sum x}{n}
\]

---

## Median

Middle value after sorting.

---

## Mode

Most frequent value.

---

# Measures of Spread

## Range

\[
\text{Range}
=
\text{Max} - \text{Min}
\]

---

## Variance

Measures dispersion from mean.

\[
\sigma^2
\]

---

## Standard Deviation

Square root of variance.

\[
\sigma
\]

---

## Interquartile Range (IQR)

\[
IQR = Q3 - Q1
\]

Where:

- Q1 = 25th Percentile
- Q2 = 50th Percentile (Median)
- Q3 = 75th Percentile

---

# Skewness, Quartiles and Box Plots

## Positive Skew

Right Tail Longer

Relationship:

```
Mode < Median < Mean
```

---

## Negative Skew

Left Tail Longer

Relationship:

```
Mean < Median < Mode
```

---

## Symmetric Distribution

```
Mean = Median = Mode
```

---

# Gaussian Distribution

Also called:

- Normal Distribution
- Bell Curve

Properties:

- Symmetric
- Mean = Median = Mode

Examples:

- Heights
- Exam Scores
- Measurement Errors

---

# Covariance

Measures relationship between two variables.

## Positive Covariance

Both increase together.

---

## Negative Covariance

One increases while the other decreases.

---

## Zero Covariance

No linear relationship.

---

# Correlation

Standardized measure of relationship.

Range:

\[
-1 \le r \le 1
\]

Values:

- +1 → Perfect Positive Relationship
- 0 → No Relationship
- -1 → Perfect Negative Relationship

Applications:

- Feature Selection
- Data Analysis
- Machine Learning

---