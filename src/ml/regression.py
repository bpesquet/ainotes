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
# # End-to-end regression example: predict housing prices
#
# > This content is heavily inspired by chapter 2 of the book [Hands-On Machine Learning](https://github.com/ageron/handson-ml2).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "slide"}
import platform

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn

print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")

# %% slideshow={"slide_type": "slide"}
# sklearn does not automatically import its subpackages
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
# Note: these configuration lines work better in their own cell

# Include matplotlib graphs into the notebook, next to the code
# https://stackoverflow.com/a/43028034/2380880
# %matplotlib inline

# Increase default plot size
# https://matplotlib.org/stable/users/explain/customizing.html#matplotlibrc-sample
plt.rcParams["figure.figsize"] = 10, 7.5

# Improve plot quality
# %config InlineBackend.figure_format = "retina"

# Setup seaborn default theme
# http://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
sns.set_theme()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Step 1: frame the problem

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
# - Slightly modified for teaching purposes by A. Géron ([original version](https://www.dcc.fc.up.pt/%7Eltorgo/Regression/cal_housing.html)).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 2.1: discover data
#
# Our first objective is to familiarize ourselves with the dataset.
#
# Once data is loaded, the [pandas](https://pandas.pydata.org/) library  provides many useful functions for making sense of it.

# %% slideshow={"slide_type": "slide"}
# Load the dataset in a pandas DataFrame
# The df_ prefix is used to distinguish dataframes froms plain NumPy arrays
dataset_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
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
df_housing.hist(bins=50, figsize=(12, 8))
plt.show()

# %% slideshow={"slide_type": "slide"}
# This dataset has the particularity of including geographical coordinates
# Visualise prices relative to them
df_housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=df_housing["population"] / 100,
    label="population",
    figsize=(12, 9),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    sharex=False,
)
plt.legend()
plt.show()

# %% slideshow={"slide_type": "slide"}
# Compute pairwise correlations of attributes
corr_matrix = df_housing.corr(numeric_only=True)

corr_matrix["median_house_value"].sort_values(ascending=False)


# %% slideshow={"slide_type": "skip"} tags=["hide-input"]
# Plot correlation matrix
def plot_correlation_matrix(df):
    # Select numerical columns
    df_numerical = df.select_dtypes(include=[np.number])

    f, ax = plt.subplots()
    ax = sns.heatmap(
        df.corr(numeric_only=True),
        vmax=0.8,
        linewidths=0.01,
        square=True,
        annot=True,
        linecolor="white",
        xticklabels=df_numerical.columns,
        annot_kws={"size": 13},
        yticklabels=df_numerical.columns,
    )


# %% slideshow={"slide_type": "slide"}
plot_correlation_matrix(df_housing)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 2.3: split the dataset
#
# Once trained, a model must be able to **generalize** (perform well with new data). In order to assert this ability, data is always split into 2 or 3 sets at the begging of the preparation phase:
#
# - **Training set** (typically 80% or more): fed to the model during training.
# - **Validation set**: used to tune the model without biasing it in favor of the test set.
# - **Test set**: used to check the final model's performance on unseen data.
#
# ![Dataset splitting](_images/dataset_splitting.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# A simple solution for splitting datasets is to use the `train_test_split`function from scikit-learn.
#
# Just before or after that, inputs (features given to the model) have to be separated from targets (values it must predict).
#
# ![Using train_test_split](_images/train-test-split.jpg)

# %% slideshow={"slide_type": "slide"}
# Separate inputs from targets

# Target attribute is removed from dataset
df_x = df_housing.drop("median_house_value", axis=1)

# Targets are stored separately in a new variable
df_y = df_housing["median_house_value"]

print(f"Inputs: {df_x.shape}")
print(f"Targets: {df_y.shape}")

# %% slideshow={"slide_type": "slide"}
# Split dataset between training and test sets (no validation set for now)
# A unique call to train_test_split is mandatory to maintain inputs/target correspondance between samples
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
df_x_train, df_x_test, df_y_train, df_y_test = sklearn.model_selection.train_test_split(df_x, df_y, test_size=0.2)

print(f"Training inputs: {df_x_train.shape}, training targets: {df_y_train.shape}")
print(f"Test inputs: {df_x_test.shape}, test targets: {df_y_test.shape}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 2.4: data preprocessing
#
# This task typically involves:
#
# - Removing of superflous features (if any).
# - Adding missing values.
# - Transforming values into numeric form.
# - Scaling data.
# - Labelling (if needed).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Handling missing values
#
# Most ML algorithms cannot work with missing values in features.
#
# Three options exist:
# - Remove the corresponding data samples.
# - Remove the whole feature(s).
# - Replace the missing values (using 0, the mean, the median or something else).

# %% slideshow={"slide_type": "slide"}
# Compute number and percent of missing values among features
total = df_x_train.isnull().sum().sort_values(ascending=False)
percent = (df_x_train.isnull().sum() * 100 / df_x_train.isnull().count()).sort_values(
    ascending=False
)
df_missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
df_missing_data.head()

# %% slideshow={"slide_type": "slide"}
# Show the first samples with missing values
df_x_train[df_x_train.isnull().any(axis=1)].head()

# %% slideshow={"slide_type": "slide"}
# Replace missing values with column-wise mean
inputer = SimpleImputer(strategy="mean")
inputer.fit_transform([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Feature scaling
#
# Most ML models work best when all features have a similar scale.
#
# One way to achieve this result is to apply **standardization**, the process of centering and reducing features: they are substracted by their mean and divided by their standard deviation.
#
# The resulting features have a mean of 0 and a standard deviation of 1.

# %% slideshow={"slide_type": "slide"}
# Generate a random (3,3) tensor with values between 1 and 10
x = np.random.randint(1, 10, (3, 3))
print(x)

# %% slideshow={"slide_type": "-"}
# Center and reduce data
scaler = StandardScaler().fit(x)
print(scaler.mean_)

x_scaled = scaler.transform(x)
print(x_scaled)

# New mean is 0. New standard deviation is 1
print(f"Mean: {x_scaled.mean()}. Std: {x_scaled.std()}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### One-hot encoding
#
# ML models expect data to be exclusively under numerical form.
#
# **One-hot encoding** produces a matrix of binary vectors from a vector of categorical values.
#
# it is useful to convert categorical features into numerical features without using arbitrary integer values, which could create a proximity relationship between the new values.

# %% slideshow={"slide_type": "slide"}
# Create a categorical variable with 3 different values
x = [["GOOD"], ["AVERAGE"], ["GOOD"], ["POOR"], ["POOR"]]

# Encoder input must be a matrix
# Output will be a sparse matrix where each column corresponds to one possible value of one feature
encoder = OneHotEncoder().fit(x)
x_hot = encoder.transform(x).toarray()

print(x_hot)
print(encoder.categories_)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Preprocessing pipelines
#
# Data preprocessing is done through a series of sequential operations on data (handling missing values, standardization, one-hot encoding...).
#
# scikit-learn support the definition of **pipelines** for streamlining these operations.

# %% slideshow={"slide_type": "slide"}
# Print numerical features
num_features = df_x_train.select_dtypes(include=[np.number]).columns
print(num_features)

# Print categorical features
cat_features = df_x_train.select_dtypes(include=[object]).columns
print(cat_features)

# Print all values for the "ocean_proximity" feature
df_x_train["ocean_proximity"].value_counts()

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

# Apply the last pipeline on training data
x_train = full_pipeline.fit_transform(df_x_train)

# Print preprocessed data shape and first sample
# "ocean_proximity" attribute has 5 different values
# To represent them, one-hot encoding has added 4 features to the dataset
print(f"x_train: {x_train.shape}")
print(x_train[0])

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
# Model performance is assessed through an **evaluation metric**.
#
# It quantifies the difference (often called **error**) between the expected results (*ground truth*) and the actual results computed by the model.
#
# A classic evaluation metric for regression tasks is the **Root Mean Square Error (RMSE)**. It gives an idea of how much error the model typically makes in its predictions.

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


# Return RMSE for a model and a training set
def compute_error(model, x, y_true):
    # Compute model predictions (median house prices) for training set
    y_pred = model.predict(x)

    # Compute the error between actual and expected median house prices
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


lin_rmse = compute_error(lin_model, x_train, y_train)
print(f"Training error for linear regression: {lin_rmse:.02f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 3.3: try other models

# %%
# Fit a decision tree model to the training set
dt_model = DecisionTreeRegressor()
dt_model.fit(x_train, y_train)

dt_rmse = compute_error(dt_model, x_train, y_train)
print(f"Training error for decision tree: {dt_rmse:.02f}")


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Using a validation set
#
# The previous result (*error = 0*) looks too good to be true. It might very well be a case of severe overfitting to the training set, which means the model won't perform well with unseen data.
#
# One way to assert overfitting is to split training data between a smaller training set and a **validation set**, used only to evaluate model performance.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Cross-validation
#
# A more sophisticated strategy is to apply **K-fold cross validation**. Training data is randomly split into $K$ subsets called *folds*. The model is trained and evaluated $K$ times, using a different fold for validation.
#
# ![K-fold Cross Validation](_images/k-fold-cross-validation.png)


# %% slideshow={"slide_type": "slide"}
# Return the mean of cross validation scores for a model and a training set
def compute_crossval_mean_score(model, x, y_true):
    scores = cross_val_score(model, x, y_true, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean()


lin_crossval_mean = compute_crossval_mean_score(lin_model, x_train, y_train)
print(f"Mean CV error for linear regression: {lin_crossval_mean:.02f}")

dt_crossval_mean = compute_crossval_mean_score(dt_model, x_train, y_train)
print(f"Mean CV error for decision tree: {dt_crossval_mean:.02f}")

# %% slideshow={"slide_type": "slide"}
# Fit a random forest model to the training set
rf_model = RandomForestRegressor(n_estimators=20)
rf_model.fit(x_train, y_train)

rf_rmse = compute_error(rf_model, x_train, y_train)
print(f"Training error for random forest: {rf_rmse:.02f}")

rf_crossval_mean = compute_crossval_mean_score(rf_model, x_train, y_train)
print(f"Mean CV error for random forest: {rf_crossval_mean:.02f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Step 4: tune the chosen model

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Searching for the best hyperparameters
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
# ### Checking final model performance on test dataset
#
# Now is the time to evaluate the final model on the test set that we put apart before.
#
# An important point is that preprocessing operations should be applied to test data using preprocessing values (mean, categories...) previously computed on training data. This prevents **information leakage** from test data ([explanation 1](https://machinelearningmastery.com/data-leakage-machine-learning/), [explanation 2](https://stats.stackexchange.com/questions/174823/how-to-apply-standardization-normalization-to-train-and-testset-if-prediction-i))

# %% slideshow={"slide_type": "slide"}
# Split test dataset between inputs and target
df_x_test = df_test.drop("median_house_value", axis=1)
y_test = df_test["median_house_value"].to_numpy()

print(f"Test data: {df_x_test.shape}")
print(f"Test labels: {y_test.shape}")

# Apply preprocessing operations to test inputs
# Calling transform() and not fit_transform() uses preprocessing values computed on training set
x_test = full_pipeline.transform(df_x_test)

test_rmse = compute_error(final_model, x_test, y_test)
print(f"Test error for final model: {test_rmse:.02f}")

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
print(f"Predicted result: {y_new[0]:.02f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Step 5: deploy to production and maintain the system

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 5.1: saving the final model and data pipeline
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
# ### Step 5.2: deploying the model
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
# ### Example: testing the deployed model
#
# The model trained in this notebook has been deployed as a [Flask web app](https://housing-prices-api.herokuapp.com/).
#
# You may access it using the web app or through a direct HTTP request to its prediction API.
#
# More details [here](https://github.com/bpesquet/housing_prices_api).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Step 5.3: monitor and maintain the system
#
# In order to guarantee an optimal quality of service, the deployed system must be carefully monitored. This may involve:
#
# * Checking the system’s live availability and performance at regular intervals.
# * Sampling the system’s predictions and evaluating them.
# * Checking input data quality.
# * Retraining the model on fresh data.
