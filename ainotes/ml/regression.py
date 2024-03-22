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
# # Regression example: predict housing prices
#
# > This chapter is inspired by the book [Hands-On Machine Learning](https://github.com/ageron/handson-ml2) written by Aurélien Géron.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Learning objectives
#
# - Discover how to train a Machine Learning model on tabular data.
# - Get familiar with the general workflow of a supervised ML project.
# - Learn how to leverage the [pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org) libraries.
#
# > You may test the trained model [here](https://housing-prices-api.herokuapp.com/).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "slide"}
import platform

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# For saving models and pipelines to disk
import joblib


# %% slideshow={"slide_type": "slide"}
# Setup plots

# Include matplotlib graphs into the notebook, next to the code
# https://stackoverflow.com/a/43028034/2380880
# %matplotlib inline

# Improve plot quality
# %config InlineBackend.figure_format = "retina"

# Setup seaborn default theme
# http://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
sns.set_theme()


# %% slideshow={"slide_type": "slide"}
# Print environment info
print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")


# %% [markdown] slideshow={"slide_type": "slide"}
# ## The Machine Learning project workflow

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Main steps of a supervised learning project
#
# 1. **Frame** the problem.
# 1. Collect, analyze and prepare **data**.
# 1. Select and train several **models** on data.
# 1. **Tune** the most promising model.
# 1. **Deploy** the model to production and monitor it.
#
# [![ML workflow by RedHat](_images/wiidii_ml_workflow.png)](https://www.redhat.com/files/summit/session-assets/2019/T957A0.pdf)
#

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Step 1: frame the problem

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Key questions
#
# - What is the business objective?
# - How good are the current solutions?
# - What data is available?
# - Is the problem a good fit for ML?
# - What is the expected learning type (supervised or not, batch/online...)?
# - How will the model's performance be evaluated?

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Properties of ML-friendly problems
#
# - Difficulty to express the actions as rules.
# - Data too complex for traditional analytical methods.
#   - High number of features.
#   - Highly correlated data (data with similar or closely related values).
# - Performance > interpretability.
# - Data quality is paramount.


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Business objective: predict housing prices
#
# - Inputs: housing properties in an area.
# - Output: median housing price in the area.
#
# [![Kaggle houses banner](_images/kaggle_housesbanner.png)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Step 2: collect, analyze and prepare data

# %% [markdown] slideshow={"slide_type": "slide"}
# ### A crucial step
#
# - Real data is messy, incomplete and often scattered across many sources.
#
# - Data labeling is a manual and tedious process.
#
# - Predefined datasets offer a convenient way to bypass the data wrangling step. Alas, using one is not always an option.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The California housing dataset
#
# - Based on data from the 1990 California census.
# - Slightly modified for teaching purposes by Aurélien Géron ([original version](https://www.dcc.fc.up.pt/%7Eltorgo/Regression/cal_housing.html)).
# - Raw CSV file is available [here](https://raw.githubusercontent.com/bpesquet/ainotes/master/data/california_housing.csv).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 2.1: discover data
#
# Our first objective is to familiarize ourselves with the dataset.
#
# Once data is loaded, the [pandas](https://pandas.pydata.org/) library  provides many useful functions for making sense of it.

# %% slideshow={"slide_type": "slide"}
# Load the dataset in a pandas DataFrame
# The df_ prefix is used to distinguish dataframes from plain NumPy arrays
dataset_url = "https://raw.githubusercontent.com/bpesquet/ainotes/master/data/california_housing.csv"
df_housing = pd.read_csv(dataset_url)

# Print dataset shape (rows and columns)
print(f"Dataset shape: {df_housing.shape}")

# %% slideshow={"slide_type": "slide"}
# Print a concise summary of the dataset
# 9 attributes are numerical, one ("ocean_proximity") is categorical
# "median_house_value" is the target attribute
# One attribute ("total_bedrooms") has missing values
df_housing.info()

# %% slideshow={"slide_type": "slide"}
# Show 10 random samples of the dataset
df_housing.sample(n=10)

# %% slideshow={"slide_type": "slide"}
# Print descriptive statistics for all numerical attributes
df_housing.describe()

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 2.2: analyze data
#
# The objective here is to gain insights about the data, in order to prepare it optimally for training.

# %% slideshow={"slide_type": "slide"}
# Plot histograms for all numerical attributes
df_housing.hist(bins=50, figsize=(10, 8))
plt.show()


# %% slideshow={"slide_type": "slide"}
def plot_geo_data(df):
    """Plot a geographical representation of the data"""

    # This dataset has the particularity of including geographical coordinates
    # Visualise prices relative to them
    df.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,
        s=df_housing["population"] / 100,
        label="population",
        figsize=(11, 8),
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=True,
        sharex=False,
    )
    plt.legend()
    plt.show()


# %%
plot_geo_data(df_housing)

# %% slideshow={"slide_type": "slide"}
# Compute pairwise correlations of attributes
corr_matrix = df_housing.corr(numeric_only=True)

corr_matrix["median_house_value"].sort_values(ascending=False)


# %% slideshow={"slide_type": "skip"} tags=["hide-input"]
def plot_correlation_matrix(df):
    """Plot a correlation matrix for a DataFrame"""

    # Select numerical columns
    df_numerical = df.select_dtypes(include=[np.number])

    plt.subplots()
    sns.heatmap(
        df.corr(numeric_only=True),
        vmax=0.8,
        linewidths=0.01,
        square=True,
        annot=True,
        linecolor="white",
        xticklabels=df_numerical.columns,
        annot_kws={"size": 10},
        yticklabels=df_numerical.columns,
    )


# %% slideshow={"slide_type": "slide"}
plot_correlation_matrix(df_housing)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 2.3: split the dataset
#
# A simple solution for splitting datasets is to use the `train_test_split`function from scikit-learn.
#
# Just before or after that, inputs (features given to the model) have to be separated from targets (values it must predict).
#
# [![Using train_test_split](_images/train-test-split.jpg)](https://mgalarnyk.github.io/)

# %% slideshow={"slide_type": "slide"}
# Separate inputs from targets

# Target attribute is removed to create inputs
df_x = df_housing.drop("median_house_value", axis="columns")

# Targets are stored separately in a new variable
df_y = df_housing["median_house_value"]

print(f"df_x: {df_x.shape}. df_y: {df_y.shape}")

# %% slideshow={"slide_type": "slide"}
# Split dataset between training and test sets
# A unique call to train_test_split is mandatory to maintain inputs/target correspondance between samples
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
    df_x, df_y, test_size=0.2
)

print(f"df_x_train: {df_x_train.shape}. df_y_train: {df_y_train.shape}")
print(f"df_x_test: {df_x_test.shape}. df_y_test: {df_y_test.shape}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 2.4: data preprocessing
#
# For this dataset, this task involves:
#
# - Handling missing values.
# - Scaling data.
# - Transforming categorical values into numeric form.

# %% slideshow={"slide_type": "slide"}
# Compute percent of missing values among features
print(df_x_train.isnull().sum() * 100 / df_x_train.isnull().count())

# %% slideshow={"slide_type": "slide"}
# Show random samples with missing values
df_x_train[df_x_train.isnull().any(axis=1)].sample(n=5)

# %% slideshow={"slide_type": "slide"}
# Get numerical features
num_features = df_x_train.select_dtypes(include=[np.number]).columns
print(num_features)

# Get categorical features
cat_features = df_x_train.select_dtypes(include=[object]).columns
print(cat_features)

# Print all values for the "ocean_proximity" feature
df_x_train["ocean_proximity"].value_counts()

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Preprocessing pipelines
#
# Data preprocessing is done through a series of sequential operations on data (handling missing values, standardization, one-hot encoding...).
#
# scikit-learn support the definition of **pipelines** for streamlining these operations. This is useful to prevent mistakes and oversights when preprocessing new data.

# %% slideshow={"slide_type": "slide"}
# This pipeline handles missing values and standardizes features
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ]
)

# This pipeline applies the previous one on numerical features
# It also one-hot encodes the categorical features
full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_features),
        ("cat", OneHotEncoder(), cat_features),
    ]
)

# %% slideshow={"slide_type": "slide"}
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

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Step 3: select and train models

# %% [markdown] slideshow={"slide_type": "slide"}
# ### An iterative and empirical step
#
# At long last, our data is ready and we can start training models.
#
# This step is often iterative and can be quite empirical. Depending on data and model complexity, it can also be resource-intensive.
#
# ![Busy training](_images/busy_training.jpg)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The optimization/generalization dilemna
#
# ![Underfitting and overfitting](_images/underfitting_overfitting.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Underfitting and overfitting
#
# - **Underfitting** (sometimes called *bias*): insufficient performance on training set.
# - **Overfitting** (sometimes called *variance*): performance gap between training and validation sets.
#
# Ultimately, we look for a tradeoff between underfitting and overfitting.
#
# The goal of the training step is to find a model powerful enough to **overfit the training set**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 3.1: choose an evaluation metric
#
# Model performance is assessed through an **evaluation metric**. Like the loss function, it depends on the problem type.
#
# A classic choice for regression tasks is the **Root Mean Square Error (RMSE)**. It gives an idea of how much error the trained model typically makes in its predictions. Of course, the smaller the better in that case.
#
# $$\mathrm{RMSE}(\pmb{Y}, \pmb{Y'}) = \sqrt{\frac{1}{m}\sum_{i=1}^m (\pmb{y'}^{(i)} - \pmb{y}^{(i)})^2}$$
#
# **Mean Absolute Error** (less sensitive to outliers) and **MSE** may also be used.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 3.2: start with a baseline model
#
# For each learning type (supervised, unsupervised...), several models of various complexity exist.
#
# It is often useful to begin the training step by using a basic model. Its results will serve as a **baseline** when training more complicated models. In some cases, its performance might even be surprisingly good.

# %% slideshow={"slide_type": "slide"}
# Fit a linear regression model to the training set
lin_model = LinearRegression()
lin_model.fit(x_train, y_train)


def compute_error(model, x, y_true):
    """Compute error (as root of MSE) for a model and a training set"""

    # Compute model predictions (median house prices) for training set
    y_pred = model.predict(x)

    # Compute the error between actual and expected median house prices
    error = np.sqrt(mean_squared_error(y_true, y_pred))
    return error


lin_error = compute_error(lin_model, x_train, y_train)
print(f"Training error for linear regression: {lin_error:.02f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 3.3: try other models

# %%
# Fit a decision tree model to the training set
dt_model = DecisionTreeRegressor()
dt_model.fit(x_train, y_train)

dt_error = compute_error(dt_model, x_train, y_train)
print(f"Training error for decision tree: {dt_error:.02f}")


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 3.4: Use a validation set to evaluate model performance
#
# The previous result (*error = 0*) looks too good to be true. It might very well be a case of severe overfitting to the training set, which means the model won't perform well with unseen data.
#
# One way to assert overfitting is to split training data between a smaller training set and a **validation set**, used only to evaluate model performance.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Cross-validation
#
# A more sophisticated strategy is to apply **K-fold cross validation**. Training data is randomly split into $K$ subsets called *folds*. The model is trained and evaluated $K$ times, using a different fold for validation.
#
# ![K-fold Cross Validation](_images/k-fold-cross-validation.png)


# %% slideshow={"slide_type": "slide"}
def compute_crossval_mean_score(model, x, y_true):
    """Return the mean of cross validation scores for a model and a training set"""

    cv_scores = -cross_val_score(
        model, x, y_true, scoring="neg_mean_squared_error", cv=10
    )
    return np.sqrt(cv_scores).mean()


lin_cv_mean = compute_crossval_mean_score(lin_model, x_train, y_train)
print(f"Mean cross-validation error for linear regression: {lin_cv_mean:.02f}")

dt_cv_mean = compute_crossval_mean_score(dt_model, x_train, y_train)
print(f"Mean cross-validation error for decision tree: {dt_cv_mean:.02f}")

# %% slideshow={"slide_type": "slide"}
# Fit a random forest model to the training set
rf_model = RandomForestRegressor(n_estimators=20)
rf_model.fit(x_train, y_train)

rf_error = compute_error(rf_model, x_train, y_train)
print(f"Training error for random forest: {rf_error:.02f}")

rf_cv_mean = compute_crossval_mean_score(rf_model, x_train, y_train)
print(f"Mean cross-validation error for random forest: {rf_cv_mean:.02f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Step 4: tune the most promising model

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 4.1: search for the best hyperparameters
#
# Once a model looks promising, it must be **tuned** in order to offer the best compromise between optimization and generalization.
#
# The goal is to find the set of model properties that gives the best performance. Model properties are often called **hyperparameters** (example: maximum depth for a decision tree).
#
# This step can be either:
#
# * manual, tweaking model hyperparameters by hand.
# * automated, using a tool to explore the model hyperparameter spaces.

# %% slideshow={"slide_type": "slide"}
# Grid search explores a user-defined set of hyperparameter values
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [50, 100, 150], "max_features": [6, 8, 10, 12]},
]

# train across 5 folds, that's a total of 12*5=60 rounds of training
grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(x_train, y_train)

# Store the best model found
final_model = grid_search.best_estimator_

# Print the best combination of hyperparameters found
print(grid_search.best_params_)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 4.2: check final model performance on test dataset
#
# Now is the time to evaluate the final model on the test set that we put apart before.
#
# An important point is that preprocessing operations should be applied to test data using preprocessing values (mean, categories...) previously computed on training data. This prevents **information leakage** from test data ([explanation](https://machinelearningmastery.com/data-leakage-machine-learning/))

# %% slideshow={"slide_type": "slide"}
# Apply preprocessing operations to test inputs
# Calling transform() and not fit_transform() uses preprocessing values computed on training set
x_test = full_pipeline.transform(df_x_test)

# Transform the targets DataFrame into a plain tensor
y_test = df_y_test.to_numpy()

test_error = compute_error(final_model, x_test, y_test)
print(f"Test error for final model: {test_error:.02f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Using the model to make predictions on new data

# %%
# Create a new data sample
new_sample = [[-122, 38, 49, 3700, 575, 1200, 543, 5.2, "NEAR BAY"]]

# Put it into a DataFrame so that it can be preprocessed
df_new_sample = pd.DataFrame(new_sample)
df_new_sample.columns = df_x_train.columns
df_new_sample.head()

# %% slideshow={"slide_type": "slide"}
# Apply preprocessing operations to new sample
# Calling transform() and not fit_transform() uses preprocessing values computed on training set
x_new = full_pipeline.transform(df_new_sample)

# Use trained model to predict median housing price
y_new = final_model.predict(x_new)
print(f"Predicted median price: {y_new[0]:.02f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Step 5: deploy to production and maintain the system

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 5.1: save the final model and data pipeline
#
# A trained model can be saved to several formats. A standard common is to use Python's built-in persistence model, [pickle](https://docs.python.org/3/library/pickle.html), through the [joblib](https://pypi.org/project/joblib/) library for efficiency reasons.

# %%
# Serialize final model and input pipeline to disk
joblib.dump(final_model, "final_model.pkl")
joblib.dump(full_pipeline, "full_pipeline.pkl")

# (Later in the process)
# model = joblib.load("final_model.pkl")
# pipeline = joblib.load("full_pipeline.pkl")
# ...

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 5.2: deploy the model
#
# This step is highly context-dependent. A deployed model is often a part of a more important system. Some common solutions:
#
# * deploying the model as a web service accessible through an API.
# * embedding the model into the user device.
#
# The [Flask](https://flask.palletsprojects.com) web framework is often used to create a web API from a trained Python model.
#
# [![Model deployement on the web](_images/model_deployment_web.png)](https://github.com/ageron/handson-ml2)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 5.3: monitor and maintain the system
#
# In order to guarantee an optimal quality of service, the deployed system must be carefully monitored. This may involve:
#
# * Checking the system’s live availability and performance at regular intervals.
# * Sampling the system’s predictions and evaluating them.
# * Checking input data quality.
# * Retraining the model on fresh data.
