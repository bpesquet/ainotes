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

# %% [markdown]
# # Heart disease
#
# ## Objective
#
# Training a model for the diagnosis of coronary artery disease (binary classification).
#
# ## Context
#
# The dataset is provided by the Cleveland Clinic Foundation for Heart Disease ([more information](https://archive.ics.uci.edu/ml/datasets/heart+Disease)). The dataset file to use is available [here](https://raw.githubusercontent.com/bpesquet/ainotes/main/data/heart.csv). Each row describes a patient. Below is a description of each column.
#
# |  Column  |                           Description                          |  Feature Type  | Data Type |
# |:--------:|:--------------------------------------------------------------:|:--------------:|:---------:|
# | Age      | Age in years                                                   | Numerical      | integer   |
# | Sex      | (1 = male; 0 = female)                                         | Categorical    | integer   |
# | CP       | Chest pain type (0, 1, 2, 3, 4)                                | Categorical    | integer   |
# | Trestbpd | Resting blood pressure (in mm Hg on admission to the hospital) | Numerical      | integer   |
# | Chol     | Serum cholestoral in mg/dl                                     | Numerical      | integer   |
# | FBS      | (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)        | Categorical    | integer   |
# | RestECG  | Resting electrocardiographic results (0, 1, 2)                 | Categorical    | integer   |
# | Thalach  | Maximum heart rate achieved                                    | Numerical      | integer   |
# | Exang    | Exercise induced angina (1 = yes; 0 = no)                      | Categorical    | integer   |
# | Oldpeak  | ST depression induced by exercise relative to rest             | Numerical      | float     |
# | Slope    | The slope of the peak exercise ST segment                      | Numerical      | integer   |
# | CA       | Number of major vessels (0-3) colored by flourosopy            | Numerical      | integer   |
# | Thal     | 3 = normal; 6 = fixed defect; 7 = reversable defect            | Categorical    | string    |
# | Target   | Diagnosis of heart disease (1 = true; 0 = false)               | Classification | integer   |
#
# ## Instructions and advice
#
# - Follow the main steps of a supervised ML project: data loading and exploring, data preparation, model training and evaluation.
# - Use the [scikit-learn](https://scikit-learn.org) library for data preparation and model training. If you are new to it, consider following its [Getting started](https://scikit-learn.org/stable/getting_started.html) guide.
# - Don't forget to setup your environment by importing the necessary Python packages.
# - Data preparation should be very similar to the [regression example](../regression.ipynb).
# - You may train any binary classification model, for example a simple [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).
# - Model evaluation should be very similar to the [classification example](../classification.ipynb).
# - Assess model performance and interpret results on test data.
# - **Bonus**: train several other models (decision tree, artificial neural network, etc) and compare their performances.

# %% [markdown]
# ## Environment setup

# %%
import platform

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn

# %%
# sklearn does not automatically import its subpackages
# https://stackoverflow.com/a/9049246/2380880
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay,
)

# %%
# Setup plots

# Include matplotlib graphs into the notebook, next to the code
# https://stackoverflow.com/a/43028034/2380880
# %matplotlib inline

# Improve plot quality
# %config InlineBackend.figure_format = "retina"

# Setup seaborn default theme
# http://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
sns.set_theme()


# %%
# Print environment info
print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")


# %% [markdown]
# ## Data loading and exploring

# %%
DATASET_URL = "https://raw.githubusercontent.com/bpesquet/ainotes/master/data/heart.csv"

df_heart = pd.read_csv(DATASET_URL)

print(f"df_heart: {df_heart.shape}")

# %%
# Print info about the dataset

df_heart.info()

# %%
# Print 10 random samples

df_heart.sample(n=10)

# %%
# Print descriptive statistics for all numerical attributes

df_heart.describe()

# %%
# Print distribution of target values

df_heart["target"].value_counts()

# %% [markdown]
# ## Data preparation
#
# ### Dataset splitting

# %%
# Separate inputs from targets

# Target attribute is removed to create inputs
df_x = df_heart.drop("target", axis="columns")

# Targets are stored separately in a new variable
df_y = df_heart["target"]

print(f"df_x: {df_x.shape}. df_y: {df_y.shape}")

# %%
# Split dataset between training and test sets
# A unique call to train_test_split is mandatory to maintain inputs/target correspondance between samples
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
    df_x, df_y, test_size=0.2
)

print(f"df_x_train: {df_x_train.shape}. df_y_train: {df_y_train.shape}")
print(f"df_x_test: {df_x_test.shape}. df_y_test: {df_y_test.shape}")

# %% [markdown]
# ### Data preprocessing

# %%
# Print numerical and categorical features

num_features = df_x_train.select_dtypes(include=[np.number]).columns
print(num_features)

cat_features = df_x_train.select_dtypes(include=[object]).columns
print(cat_features)

# %%
# Print all values for the "thal" feature

df_x_train["thal"].value_counts()

# %%
# Preprocess data to have similar scales and only numerical values

# This pipeline standardizes numerical features
# It also one-hot encodes the categorical features
full_pipeline = ColumnTransformer(
    [
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(), cat_features),
    ]
)

# %%
# Apply all preprocessing operations to the training set through pipelines
x_train = full_pipeline.fit_transform(df_x_train)

# Transform the targets DataFrame into a plain tensor
y_train = df_y_train.to_numpy()

# Print preprocessed data shape and first sample
# "ocean_proximity" attribute has 5 different values
# To represent them, one-hot encoding has added 4 features to the dataset
print(f"x_train: {x_train.shape}")
print(x_train[0])

# Data is now ready for model training :)

# %% [markdown]
# ## Model training

# %%
# Fit a SGD classifier to the training set

sgd_model = SGDClassifier(loss="log_loss")
sgd_model.fit(x_train, y_train)

# %% [markdown]
# ## Performance assessment

# %% [markdown]
# ### Cross-validation accuracy

# %%
# Use cross-validation to evaluate accuracy, using 3 folds

cv_acc = cross_val_score(sgd_model, x_train, y_train, cv=3, scoring="accuracy")

print(f"CV accuracy: {cv_acc}")


# %% [markdown]
# ### Confusion matrix


# %%
def plot_conf_mat(model, x, y):
    """Plot the confusion matrix for a model, inputs and targets"""

    with sns.axes_style("white"):  # Temporary hide Seaborn grid lines
        _ = ConfusionMatrixDisplay.from_estimator(
            model, x, y, values_format="d", cmap=plt.colormaps.get_cmap("Blues")
        )


# %%
# Plot confusion matrix for the SGD classifier

plot_conf_mat(sgd_model, x_train, y_train)

# %% [markdown]
# ### Precision, recall and ROC curve

# %%
# Compute precision, recall and f1-score for the SGD classifier

print(classification_report(y_train, sgd_model.predict(x_train)))

# %%
# Plot ROC curve for the SGD classifier

sgd_disp = RocCurveDisplay.from_estimator(sgd_model, x_train, y_train)
plt.show()

# %% [markdown]
# ### Computing metrics on test data

# %%
# Apply preprocessing operations to test inputs
# Calling transform() and not fit_transform() uses preprocessing values computed on training set
x_test = full_pipeline.transform(df_x_test)

# Transform the targets DataFrame into a plain tensor
y_test = df_y_test.to_numpy()

print(f"x_test: {x_train.shape}. y_test: {y_test.shape}")

# %%
# Compute precision, recall and f1-score for the SGD classifier

print(classification_report(y_test, sgd_model.predict(x_test)))

# %% [markdown]
# ### Results interpretation
#
# Since the dataset is skewed (less positive than negative targets), accuracy is not sufficient to assess model performance.
#
# Results might vary quite a lot from training to training. This is probably due to the small number of samples and the stochastic nature of gradient descent.
#
# Generally speaking, the model does fairly well but the recall for true targets (patients with heart disease) might not be very good. Since this is the most important metric here (false negative could turn deadly), caution should be exercised before putting the trained classifier into production.
#
# In any case, further tests should be run to evaluate other model architectures (decision trees, artificial neural networks, etc) on this task.

# %%
