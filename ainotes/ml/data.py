# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Data manipulation

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Learning objectives
#
# - Discover what tensors are and how to manipulate them with [NumPy](https://numpy.org/) and [PyTorch](https://pytorch.org/).
# - Be able to load and prepare datasets of different types (tabular data, images or videos) for training a Machine Learning model.
# - Learn how the [pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org) libraries can help achieve the previous task.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "skip"}
# pylint: disable=wrong-import-position

import os

# Installing the ainotes package is only necessary in standalone runtime environments like Colab
if os.getenv("COLAB_RELEASE_TAG"):
    print("Standalone runtime environment detected, installing ainotes package")
    # %pip install ainotes

# pylint: enable=wrong-import-position

# %% slideshow={"slide_type": "slide"}
import platform

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn
from sklearn.datasets import load_sample_images
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

import torch

from ainotes.utils.train import get_device

# %% slideshow={"slide_type": "slide"}
# Setup plots

# Include matplotlib graphs into the notebook, next to the code
# https://stackoverflow.com/a/43028034/2380880
# %matplotlib inline

# Improve plot quality
# %config InlineBackend.figure_format = "retina"

# %% slideshow={"slide_type": "slide"}
# Print environment info
print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"PyTorch version: {torch.__version__}")


# PyTorch device configuration
device, message = get_device()
print(message)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Working with tensors

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Definition
#
# In the context of AI, a **tensor** is a set of primitive values sharing the same type (most often numerical), shaped into an array of any number of dimensions. It is a fancy name for a multidimensional array.
#
# Tensors are heavily used by AI algorithms to represent and manipulate information. They are, in particular, the core data structures of Machine Learning.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Tensor properties
#
# - A tensor's dimension is also called an **axis**.
# - A tensor's **rank** is its number of axes.
# - The tensor's **shape** describes the number of values along each axis.
#
# In mathematical terms, a rank 0 tensor is a **scalar**, a rank 1 tensor is a **vector** and a rank 2 tensor is a **matrix**.
#
# > Warning: *rank* and *dimension* are polysemic terms, which can be confusing.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Tensors in Python
#
# Python offers limited native support for manipulating tensors. Lists can be used to store information, but their mathematical capacities are insufficient for any serious work.

# %%
# A vector (rank 1 tensor)
a = [1, 2, 3]
print(a)

# A matrix (rank 2 tensor)
b = [a, [4, 5, 6]]
print(b)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Dedicated libraries
#
# Over the years, several tools have been developed to overcome Python's native limitations.
#
# The most widely used is [NumPy](https://numpy.org/), which supports tensors in the form of `ndarray` objects. It offers a comprehensive set of operations on them, including creating, sorting, selecting, linear algebra and statistical operations.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Tensor management with NumPy
#
# #### Creating tensors
#
# The [np.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html) function creates and returns a new tensor.
#
# [![NumPy array creation](_images/numpy_array.png)](https://numpy.org/doc/stable/user/absolute_beginners.html#how-to-create-a-basic-array)
#
# The [NumPy API](https://numpy.org/doc/stable/reference/routines.array-creation.html) contains many functions for creating tensors using predefined values.


# %% slideshow={"slide_type": "slide"}
def print_tensor_info(t):
    """Print values, number of dimensions and shape of a tensor"""

    print(t)
    print(f"Dimensions: {t.ndim}")
    print(f"Shape: {t.shape}")


# %% slideshow={"slide_type": "-"}
# Create a scalar
x = np.array(12)

print_tensor_info(x)

# %% slideshow={"slide_type": "-"}
# Create a vector (1D tensor)
x = np.array([1, 2, 3])

print_tensor_info(x)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Generating random tensors
#
# The [NumPy API](https://numpy.org/doc/stable/reference/random/index.html) also permits the creation of (pseudo-)randomly valued tensors, using various statistical laws and data types.

# %% slideshow={"slide_type": "-"}
# Init a NumPy random number generator
rng = np.random.default_rng()

# Create a 3x4 random matrix (2D tensor) with real values sampled from a uniform distribution
x = rng.uniform(size=(3, 4))

print_tensor_info(x)

# %% slideshow={"slide_type": "slide"}
# Create a 3x2x5 3D tensor with integer values sampled from a uniform distribution
x = rng.integers(low=0, high=100, size=(3, 2, 5))

print_tensor_info(x)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Shape management
#
# A common operation on tensors is **reshaping**: giving it a new shape without changing its data.
#
# The new shape must be compatible with the current one: the new tensor needs to have the same number of elements as the original one.
#
# [![NumPy reshaping](_images/numpy_reshaping.png)](https://numpy.org/doc/stable/user/absolute_beginners.html#transposing-and-reshaping-a-matrix)

# %% slideshow={"slide_type": "slide"}
# Reshape a 3x2 matrix into a 2x3 matrix
x = np.array([[1, 2], [3, 4], [5, 6]])
x_reshaped = x.reshape(2, 3)

print_tensor_info(x_reshaped)

# %% slideshow={"slide_type": "-"}
# Reshape the previous matrix into a vector
x_reshaped = x.reshape(
    6,
)

print_tensor_info(x_reshaped)

# %% slideshow={"slide_type": "-"}
# Error: incompatible shapes!
# x.reshape(5, )

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Indexing and slicing
#
# Tensors can be indexed and sliced just like regular Python lists.
#
# [![NumPy indexing](_images/numpy_indexing.png)](https://numpy.org/doc/stable/user/absolute_beginners.html#indexing-and-slicing)

# %% slideshow={"slide_type": "slide"}
x = np.array([1, 2, 3])

# Select element at index 1
assert x[1] == 2

# Select elements between indexes 0 (included) and 2 (excluded)
assert np.array_equal(x[0:2], [1, 2])

# Select elements starting at index 1 (included)
assert np.array_equal(x[1:], [2, 3])

# Select last element
assert np.array_equal(x[-1], 3)

# Select all elements but last one
assert np.array_equal(x[:-1], [1, 2])

# Select last 2 elements
assert np.array_equal(x[-2:], [2, 3])

# Select second-to-last element
assert np.array_equal(x[-2:-1], [2])

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Tensor axes
#
# Many tensor operations can be applied along one or several **axes**. They are indexed starting at 0.
#
# [![NumPy axes](_images/numpy_axes.png)](https://www.sharpsightlabs.com/blog/numpy-axes-explained/)

# %% slideshow={"slide_type": "slide"}
# Create a 2x2 matrix (2D tensor)
x = np.array([[1, 1], [2, 2]])
print(x)

# Summing values on first axis (rows)
print(x.sum(axis=0))

# Summing values on second axis (columns)
print(x.sum(axis=1))

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Element-wise operations
#
# These operations are applied independently to each entry in the tensors being considered.

# %%
# Element-wise product between two matrices (shapes must be identical)
x = np.array([[1, 2, 3], [3, 2, -2]])
y = np.array([[3, 0, 2], [1, 4, -2]])
z = x * y

print(x)
print(y)
print_tensor_info(z)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Dot product
#
# On the contrary, operations like dot product combine entries in the input tensors to produce a differently shaped result.

# %%
# Dot product between two matrices (shapes must be compatible)
x = np.array([[1, 2, 3], [3, 2, 1]])
y = np.array([[3, 0], [2, 1], [4, -2]])
# alternative syntax: z = x.dot(y)
z = np.dot(x, y)

print(x)
print(y)
print_tensor_info(z)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Broadcasting
#
# **Broadcasting** is a mechanism that allows operations to be performed on tensors of different shapes. Subject to [certain constraints](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules), the smaller tensor may be “broadcast” across the larger one so that they have compatible shapes.
#
# [![NumPy broadcasting](_images/numpy_broadcasting.png)](https://numpy.org/doc/stable/user/absolute_beginners.html#broadcasting)
#
# Broadcasting provides a efficient means of *vectorizing* tensor operations.

# %%
# Broadcasting between a vector and a scalar
x = np.array([1.0, 2.0])
print(x * 1.6)


# %% slideshow={"slide_type": "slide"}
# Broadcasting between a matrix and a vector
x = np.array([[0, 1, 2], [-2, 5, 3]])
y = np.array([1, 2, 3])
z = x + y

print_tensor_info(z)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### GPU-based tensors
#
# For all its qualities, NumPy has a limitation which can be critical in some contexts: it only runs on the machine's [CPU](https://en.wikipedia.org/wiki/Processor_(computing)).
#
# Among other advantages, newer tools offer support for dedicated high-performance processors like [GPUs](https://en.wikipedia.org/wiki/Graphics_processing_unit) or [TPUs](https://en.wikipedia.org/wiki/Tensor_Processing_Unit), while providing a NumPy-like API to make onboarding easier. The most prominent ones are currently [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org) and [JAX](https://jax.readthedocs.io).

# %%
# Create a 2x2 random PyTorch tensor, trying to store it into the GPU memory
x = torch.rand(size=(2, 2), device=device)
print(x)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Loading and exploring data

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Introduction to pandas
#
# The [pandas](https://pandas.pydata.org/) library is dedicated to data analysis in Python. It greatly facilitates loading, exploring and processing tabular data files.
#
# The primary data structures in pandas are implemented as two classes:
#
# - `DataFrame`, which is quite similar to as a relational data table, with rows and named columns.
# - `Series`, which represents a single data column. A DataFrame contains one or more Series and a name for each Series.
#
# The DataFrame is a commonly used abstraction for data manipulation.

# %% slideshow={"slide_type": "slide"}
# Create two data Series
pop = pd.Series({"CAL": 38332521, "TEX": 26448193, "NY": 19651127})
area = pd.Series({"CAL": 423967, "TEX": 695662, "NY": 141297})

# Create a DataFrame contraining the two Series
# The df_ prefix is used to distinguish pandas dataframes from plain NumPy arrays
df_poprep = pd.DataFrame({"population": pop, "area": area})

print(df_poprep)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Loading a tabular dataset
#
# When describing tabular information, most datasets are stored as a CSV (Comma-Separated Values) text file.
#
# The [pd.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) function can load a CSV file into a DataFrame from either a local path or a URL.
#
# The following code loads a dataset which was extracted from a [Kaggle competition](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results).

# %%
# Load a CSV file into a DataFrame
# Data comes from this Kaggle competition:
df_olympics = pd.read_csv(
    "https://raw.githubusercontent.com/bpesquet/ainotes/master/data/athlete_events.csv"
)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Exploring tabular data
#
# Once a dataset is loaded into a DataFrame, many operations can be applied to it for visualization or transformation purposes. For more details, see the [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) tutorial.
#
# Let's use pandas to perform the very first steps of what is often called *Exploratory Data Analysis*.

# %%
# Print dataset shape (rows x columns)
print(f"df_olympics: {df_olympics.shape}")

# %% slideshow={"slide_type": "slide"}
# Print a concise summary of the dataset
df_olympics.info()

# %% slideshow={"slide_type": "slide"}
# Print the first 10 rows of the dataset
df_olympics.head(n=10)

# %% slideshow={"slide_type": "slide"}
# Print 5 random samples
df_olympics.sample(n=5)

# %% slideshow={"slide_type": "slide"}
# Print descriptive statistics for all numerical attributes
df_olympics.describe()

# %% slideshow={"slide_type": "slide"}
# Print descriptive statistics for all non-numerical attributes
df_olympics.describe(include=["object", "bool"])

# %% slideshow={"slide_type": "slide"}
# Print the number of samples by sport
df_olympics["Sport"].value_counts()

# %% slideshow={"slide_type": "slide"}
# Print rounded average age for women
age_mean = df_olympics[df_olympics["Sex"] == "F"]["Age"].mean()
print(f"{age_mean:.0f}")

# %% slideshow={"slide_type": "-"}
# Print percent of athletes for some countries

athletes_count = df_olympics.shape[0]

for country in ["USA", "FRA", "GBR", "GER"]:
    percent = (df_olympics["NOC"] == country).sum() / athletes_count
    print(f"Athletes from {country}: {percent*100:.2f}%")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Loading images
#
# Digital images are stored using either the **bitmap** format (an array of color values for all individual pixels in the image) or the **vector format** (a description of the elementary shapes in the image).
#
# Bitmap images can be easily manipulated as tensors. Each pixel color is typically expressed using a combination of the three primary colors: red, green and blue.
#
# [![RGB wheel](_images/rgb_wheel.jpg)](https://medium.com/@brugmanj/coding-and-colors-a-practical-approach-to-hex-and-rgb-values-9a6e98720b25)

# %% slideshow={"slide_type": "slide"}
# Load sample images provided by scikit-learn into a NumPy array
images = np.asarray(load_sample_images().images)

# Load the last sample image
sample_image = images[-1]

# Show image
plt.imshow(sample_image)

print(f"Images: {images.shape}")
print(f"Sample image: {sample_image.shape}")
print(f"Sample pixel: {sample_image[225, 300]}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Preparing data for training

# %% [markdown] slideshow={"slide_type": "slide"}
# ### A mandatory step
#
# In Machine Learning, the chosen dataset has to be carefully prepared before using it to train a model. This can have a major impact on the outcome of the training process.
#
# This important task, sometimes called *data preprocessing*, might involve:
#
# - Splitting data between training, validation and test sets.
# - Reshaping data.
# - Removing superflous features (if any).
# - Adding missing values.
# - Scaling data.
# - Transforming values into numeric form.
# - Augmenting data.
# - Engineering new features.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Data splitting
#
# Once trained, a ML model must be able to **generalize** (perform well with new data). In order to assert this ability, data is always split into two or three sets before training:
#
# - **Training set** (typically 80% or more): fed to the model during training.
# - **Validation set**: used to tune the model without biasing it in favor of the test set.
# - **Test set**: used to check the final model's performance on unseen data.
#
# ![Dataset splitting](_images/dataset_splitting.png)

# %% slideshow={"slide_type": "slide"}
# Demonstrate the use of scikit-learn's train_test_split for splitting a dataset

# Create a random 30x4 matrix (fictitious inputs) and a random 30x1 vector (fictitious results)
x = np.random.rand(30, 4)
y = np.random.rand(30)
print(f"x: {x.shape}. y: {y.shape}")

# Split fictitious dataset between training and test sets, using a 75/25 ratio
# A unique call to train_test_split is mandatory to maintain inputs/target correspondance between samples
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print(f"x_train: {x_train.shape}. y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}. y_test: {y_test.shape}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Image and video reshaping
#
# A bitmap image can be represented as a 3D multidimensional array of dimensions $height \times width \times color\_channels$.
#
# A video can be represented as a 4D multidimensional array of dimensions $frames \times height \times width \times color\_channels$.
#
# They have to be **reshaped**, or more precisely *flattened* in that case, into a vector (1D tensor) before being fed to most ML algorithms.

# %% [markdown] slideshow={"slide_type": "slide"}
# ![Reshaping an image](_images/image2vector.jpeg)

# %% slideshow={"slide_type": "slide"}
# Flatten the image, which is a 3D tensor, into a vector (1D tensor)
flattened_image = sample_image.reshape((427 * 640 * 3,))

# Alternative syntaxes to achieve the same result
# -1 means the new dimension is inferred from current dimensions
# Diference between flatten() and ravel() is explained here:
# https://numpy.org/doc/stable/user/absolute_beginners.html#reshaping-and-flattening-multidimensional-arrays
flattened_image = sample_image.reshape((-1,))
flattened_image = sample_image.ravel()
flattened_image = sample_image.flatten()

print(f"Flattened image: {flattened_image.shape}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Handling of missing values
#
# Most ML algorithms cannot work with missing values in features.
#
# Depending on the percentage of missing data, three options exist:
# - remove the corresponding data samples;
# - remove the whole feature(s);
# - replace the missing values (using 0, the mean, the median or something more meaningful in the context).

# %% slideshow={"slide_type": "slide"}
# Demonstrate the use of scikit-learn's SimpleImputer for handling missing values

# Create a tensor with missing values
x = np.array([[7, 2, np.nan], [4, np.nan, 6], [10, 5, 9]])
print(x)

# Replace missing values with column-wise mean
imputer = SimpleImputer(strategy="mean")
print(imputer.fit_transform(x))

# Replace missing values with "Unknown"
imputer = SimpleImputer(strategy="constant", missing_values=None, fill_value="Unknown")
print(imputer.fit_transform([["M"], ["M"], [None], ["F"], [None]]))

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Feature scaling
#
# Most ML algorithms work best when all features have a **similar scale**. Several solutions exist:
#
# - **Min-Max scaling**: features are shifted and rescaled to the $[0,1]$ range by substracting the `min` value and dividing by `(max-min)` on the first axis.
# - **Standardization**: features are centered (substracted by their mean) then reduced (divided by their standard deviation) on the first axis. All resulting features have a mean of 0 and a standard deviation of 1.

# %% slideshow={"slide_type": "slide"}
# Demonstrate the use of scikit-learn's MinMaxScaler to rescale values

# Generate a random 3x4 tensor with integer values between 1 and 10
x = np.random.randint(1, 10, (3, 4))
print(x)

# Compute min and max then scale tensor in one operation
x_scaled = MinMaxScaler().fit_transform(x)

print(x_scaled)
print(f"Minimum: {x_scaled.min(axis=0)}. Maximum: {x_scaled.max(axis=0)}")

# %% slideshow={"slide_type": "slide"}
# Demonstrate the use of scikit-learn's StandardScaler to standardize values

# Generate a random (3,4) tensor with integer values between 1 and 10
x = np.random.randint(1, 10, (3, 4))
print(x)

# Center and reduce data
scaler = StandardScaler().fit(x)
print(scaler.mean_)

x_scaled = scaler.transform(x)
print(x_scaled)

# New mean is 0. New standard deviation is 1
print(f"Mean: {x_scaled.mean()}. Std: {x_scaled.std()}")

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Feature scaling and training/test sets
#
# In order to avoid [information leakage](https://stats.stackexchange.com/a/174865), the test set must be scaled with values calculated on the training set.

# %% slideshow={"slide_type": "-"}
# Compute mean and std on training set
scaler = StandardScaler().fit(x_train)

# Standardize training and test sets, using mean and std computed on training set
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"Train mean: {x_train_scaled.mean(axis=0)}")
print(f"Train std: {x_train_scaled.std(axis=0)}")
print(f"Test mean: {x_test_scaled.mean(axis=0)}")
print(f"Test std: {x_test_scaled.std(axis=0)}")

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Image and video scaling
#
# Individual pixel values for images and videos are typically integers in the $[0,255]$ range. This is not ideal for most ML algorithms.
#
# Dividing them by $255.0$ to obtain floats into the $[0,1]$ range is a common practice.

# %% slideshow={"slide_type": "-"}
# Scaling sample image pixels between 0 and 1
scaled_image = sample_image / 255.0

# Check that all values are in the [0,1] range
assert scaled_image.min() >= 0
assert scaled_image.max() <= 1

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Encoding of categorical features
#
# Some features or targets may come as discrete rather than continuous values. Moreover, these discrete values might be strings. ML models are only able to manage numerical-only data.
#
# A solution is to apply one-of-K encoding, also named **dummy encoding** or **one-hot encoding**. Each categorical feature with `K` possible values is transformed into a vector of `K` binary features, with one of them 1 and all others 0.
#
# Note: using arbitrary integer values rather than binary vectors would create a proximity relationship between the new features, which could confuse the model during training.

# %% slideshow={"slide_type": "slide"}
# Demonstrate the use of scikit-learn's OneHotEncoder to one-hot encode categorical features

# Create a categorical variable with 3 different values
x = [["GOOD"], ["AVERAGE"], ["GOOD"], ["POOR"], ["POOR"]]

# Encoder input must be a matrix
# Output will be a sparse matrix where each column corresponds to one possible value of one feature
encoder = OneHotEncoder().fit(x)
x_hot = encoder.transform(x).toarray()

print(x_hot)
print(encoder.categories_)

# %% slideshow={"slide_type": "slide"}
# Demonstrate one-hot encoding of categorical features given as integers

# Generate a (5,1) tensor with integer values between 0 and 9
x = np.random.randint(0, 9, (5, 1))
print(x)

# Encoder input must be a matrix
# Output will be a sparse matrix where each column corresponds to one possible value of one feature
x_hot = OneHotEncoder().fit_transform(x).toarray()

print(x_hot)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### One-hot encoding and training/test sets
#
# Depending on value distribution between training and test sets, some categories might appear only in one set.
#
# The best solution is to one-hot encode based on the training set categories, ignoring test-only categories.

# %% slideshow={"slide_type": "slide"}
x_train = [["Blue"], ["Red"], ["Blue"], ["Green"]]
# "Yellow" is not present in training set
x_test = [
    ["Red"],
    ["Yellow"],
    ["Green"],
    ["Yellow"],
]

# Using categories from train set, ignoring unkwown categories
encoder = OneHotEncoder(handle_unknown="ignore").fit(x_train)
print(encoder.transform(x_train).toarray())
print(encoder.categories_)

# Unknown categories will result in a binary vector will all zeros
print(encoder.transform(x_test).toarray())

# %%
x = [["M"], ["M"], [None], ["F"]]

# Replace missing values with constant
print(
    SimpleImputer(
        strategy="constant", missing_values=None, fill_value="Unknown"
    ).fit_transform(x)
)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Data augmentation
#
# **Data augmentation** is the process of enriching a dataset by adding new samples, slightly modified copies of existing data or newly created synthetic data.
#
# [![Image augmentation example](_images/image_augmentation.png)](https://towardsdatascience.com/machinex-image-data-augmentation-using-keras-b459ef87cd22)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Feature engineering
#
# **Feature engineering** is the process of preparing the proper input features, in order to facilitate the learning task. The problem is made easier by expressing it in a simpler way. This usually requires a good domain knowledge.
#
# The ability of deep neural networks to discover useful features by themselves has somewhat reduced the criticality of feature engineering. Nevertheless, it remains important in order to solve problems more elegantly and with fewer data.

# %% [markdown] slideshow={"slide_type": "slide"}
# Example (taken from the book [Deep Learning with Python](https://github.com/fchollet/deep-learning-with-python-notebooks)): the task of learning the time of day from a clock is far easier with engineered features rather than raw clock images.
#
# [![Feature engineering](_images/feature_engineering.png)](https://www.manning.com/books/deep-learning-with-python)

# %%
