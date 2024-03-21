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

# %%
