# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Data preprocessing

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Objectives
#
# - Being able to load and prepare a dataset (tabular data, images or videos) for training a Machine Learning model.
# - Learn how the [scikit-learn](https://scikit-learn.org) library can simplify this task.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "-"}
# Relax some linting rules not needed here
# pylint: disable=invalid-name,wrong-import-position

import platform

import numpy as np
import matplotlib.pyplot as plt
import sklearn

print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")

# %% slideshow={"slide_type": "slide"}
# sklearn does not automatically import its subpackages
# https://stackoverflow.com/a/9049246/2380880
from sklearn.datasets import load_sample_images
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


# %% slideshow={"slide_type": "slide"}
# Setup plots

# Include matplotlib graphs into the notebook, next to the code
# https://stackoverflow.com/a/43028034/2380880
# %matplotlib inline

# Increase default plot size
# https://matplotlib.org/stable/users/explain/customizing.html#matplotlibrc-sample
plt.rcParams["figure.figsize"] = 10, 7.5

# Improve plot quality
# %config InlineBackend.figure_format = "retina"


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Image and video reshaping
#
# A bitmap image can be represented as a 3D multidimensional array of dimensions $height \times width \times color\_channels$.
#
# A video can be represented as a 4D multidimensional array of dimensions $frames \times height \times width \times color\_channels$.
#
# They have to be **reshaped**, or more precisely *flattened* in that case, into a vector (1D tensor) before being fed to most ML algorithms.

# %% [markdown] slideshow={"slide_type": "slide"}
# ![Reshaping an image](_images/image2vector.jpeg)

# %% slideshow={"slide_type": "slide"}
# Load a sample image provided by scikit-learn
sample_image = np.asarray(load_sample_images().images)[1]

# Show image
plt.imshow(sample_image)

print(f"Sample image: {sample_image.shape}")

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
# ## Dataset splitting
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
# ## Handling of missing values
#
# Most ML algorithms cannot work with missing values in features.
#
# Depending on the percentage of missing data, three options exist:
# - remove the corresponding data samples;
# - remove the whole feature(s);
# - replace the missing values (using 0, the mean, the median or something more meaningful in the context).

# %% slideshow={"slide_type": "slide"}
# Demonstrate the use of scikit-learn's SimpleImputer for handling missing values

# Replace missing values with column-wise mean
imputer = SimpleImputer(strategy="mean")
print(imputer.fit_transform([[7, 2, np.nan], [4, np.nan, 6], [10, 5, 9]]))

# Replace missing values with "Unknown"
imputer = SimpleImputer(strategy="constant", missing_values=None, fill_value="Unknown")
print(imputer.fit_transform([["M"], ["M"], [None], ["F"], [None]]))

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Feature scaling
#
# Most ML algorithms work best when all features have a **similar scale**. Several solutions exist:
#
# - **Min-Max scaling**: features are shifted and rescaled to the $[0,1]$ range by substracting the `min` value and dividing by `(max-min)` on the first axis.
# - **Standardization**: features are centered (substracted by their mean) then reduced (divided by their standard deviation) on the first axis. All resulting features have a mean of 0 and a standard deviation of 1.
#
# In order to avoid [information leakage](https://stats.stackexchange.com/a/174865), the test set must be scaled with values calculated on the training set.

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

# %% slideshow={"slide_type": "slide"}
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
# ### Image and video scaling
#
# Individual pixel values for images and videos are typically integers in the $[0,255]$ range. This is not ideal for most ML algorithms.
#
# Dividing them by 255.0 to obtain floats into the $[0,1]$ range is a common practice.

# %% slideshow={"slide_type": "-"}
# Scaling sample image pixels between 0 and 1
scaled_image = sample_image / 255.0

# Check that all values are in the [0,1] range
assert scaled_image.min() >= 0
assert scaled_image.max() <= 1

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Encoding of categorical features
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
# Generate a (5,1) tensor with integer values between 0 and 9
x = np.random.randint(0, 9, (5, 1))
print(x)

# Encoder input must be a matrix
# Output will be a sparse matrix where each column corresponds to one possible value of one feature
x_hot = OneHotEncoder().fit_transform(x).toarray()

print(x_hot)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### One-hot encoding and training/test sets
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
# ## Data augmentation
#
# **Data augmentation** is the process of enriching a dataset by adding new samples, slightly modified copies of existing data or newly created synthetic data.
#
# [![Image augmentation example](_images/image_augmentation.png)](https://towardsdatascience.com/machinex-image-data-augmentation-using-keras-b459ef87cd22)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Feature engineering
#
# **Feature engineering** is the process of preparing the proper input features, in order to facilitate the learning task. The problem is made easier by expressing it in a simpler way. This usually requires a good domain knowledge.
#
# The ability of deep neural networks to discover useful features by themselves has somewhat reduced the criticality of feature engineering. Nevertheless, it remains important in order to solve problems more elegantly and with fewer data.

# %% [markdown] slideshow={"slide_type": "slide"}
# Example: the task of learning the time of day from a clock is far easier with engineered features rather than raw clock images.
#
# [![Feature engineering](_images/feature_engineering.png)](https://www.manning.com/books/deep-learning-with-python)

# %%
