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
# # Classification example: recognize handwritten digits
#
# > This chapter is inspired by the book [Hands-On Machine Learning](https://github.com/ageron/handson-ml2) written by Aurélien Géron.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Learning objectives
#
# - Discover how to train a Machine Learning model on bitmap images.
# - Understand how loss and model performance are evaluated in classification tasks.
# - Discover several performance metrics and how to choose between them.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "slide"}
import platform

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay,
    log_loss,
)
from sklearn.linear_model import SGDClassifier

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
# ## Context and data preparation

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The MNIST handwritten digits dataset
#
# This [dataset](http://yann.lecun.com/exdb/mnist/), a staple of Machine Learning and the "Hello, world!" of computer vision, contains 70,000 bitmap images of digits.
#
# The associated target (expected result) for any image is the digit its represents.

# %% slideshow={"slide_type": "-"}
# Load the MNIST digits dataset from sciki-learn
images, targets = fetch_openml(
    "mnist_784", version=1, parser="pandas", as_frame=False, return_X_y=True
)

print(f"Images: {images.shape}. Targets: {targets.shape}")
print(f"First 10 labels: {targets[:10]}")

# %% slideshow={"slide_type": "slide"}
# Show raw data for the first digit image
print(images[0])

# %% slideshow={"slide_type": "slide"}
# Plot the first 10 digits

# Temporary hide Seaborn grid lines
with sns.axes_style("white"):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        digit = images[i].reshape(28, 28)
        fig = plt.subplot(2, 5, i + 1)
        plt.imshow(digit)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Training and test sets
#
# Data preparation begins with splitting the dataset between training and test sets.

# %% slideshow={"slide_type": "-"}
# Split dataset into training and test sets
train_images, test_images, train_targets, test_targets = train_test_split(
    images, targets, test_size=10000
)

print(f"Training images: {train_images.shape}. Training targets: {train_targets.shape}")
print(f"Test images: {test_images.shape}. Test targets: {test_targets.shape}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Images rescaling
#
# For grayscale bitmap images, each pixel value is an integer between $0$ and $255$.
#
# Next, we need to rescale pixel values into the $[0,1]$ range. The easiest way is to divide each value by $255.0$.

# %%
# Rescale pixel values from [0,255] to [0,1]
x_train, x_test = train_images / 255.0, test_images / 255.0

print(f"x_train: {x_train.shape}")
print(f"x_test: {x_train.shape}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Binary classification

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Creating binary targets
#
# To simplify things, let's start by trying to identify one digit: the number 5. The problem is now a **binary classification** task.

# %%
# Transform results into binary values
# label is true for all 5s, false for all other digits
y_train_5 = train_targets == "5"
y_test_5 = train_targets == "5"

print(train_targets[:10])
print(y_train_5[:10])

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Choosing a loss function
#
# This choice depends on the problem type. For binary classification tasks where expected results are either 1 (*True*) or 0 (*False*), a popular choice is the **Binary Cross Entropy loss**, a.k.a. **log(istic regression) loss**. It is implemented in the scikit-learn [log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) function.
#
# $$\mathcal{L}_{\mathrm{BCE}}(\pmb{\omega}) = -\frac{1}{m}\sum_{i=1}^m \left(y^{(i)} \log_e(y'^{(i)}) + (1-y^{(i)}) \log_e(1-y'^{(i)})\right)$$
#
# - $y^{(i)} \in \{0,1\}$: expected result for the $i$th sample.
# - $y'^{(i)} = h_{\pmb{\omega}}(\pmb{x}^{(i)}) \in [0,1]$: model output for the $i$th sample, i.e. probability that the $i$th sample belongs to the positive class.


# %% slideshow={"slide_type": "slide"}
def plot_bce():
    """Plot BCE loss for one output"""

    x = np.linspace(0.01, 0.99, 200)
    plt.plot(x, -np.log(1 - x), label="Target = 0")
    plt.plot(x, -np.log(x), "r--", label="Target = 1")
    plt.xlabel("Model output")
    plt.ylabel("Loss value")
    plt.legend(fontsize=12)
    plt.show()


# %% slideshow={"slide_type": "slide"}
plot_bce()

# %% slideshow={"slide_type": "slide"}
# Compute BCE losses for pseudo-predictions

y_true = [0, 0, 1, 1]

# Good prediction
y_pred = [0.1, 0.2, 0.7, 0.99]
bce = log_loss(y_true, y_pred)
print(f"BCE loss (good prediction): {bce:.05f}")

# Compare theorical and computed values
np.testing.assert_almost_equal(
    -(np.log(0.9) + np.log(0.8) + np.log(0.7) + np.log(0.99)) / 4, bce, decimal=5
)

# Perfect prediction
y_pred = [0.0, 0.0, 1.0, 1.0]
print(f"BCE loss (perfect prediction): {log_loss(y_true, y_pred):.05f}")

# Awful prediction
y_pred = [0.9, 0.85, 0.17, 0.05]
print(f"BCE loss (awful prediction): {log_loss(y_true, y_pred):.05f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Training a binary classifier

# %%
# Create a classifier using stochastic gradient descent and logistic loss
sgd_model = SGDClassifier(loss="log_loss")

# Train the model on data
sgd_model.fit(x_train, y_train_5)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Assesing  performance
#
# #### Thresholding model output
#
# A ML model computes probabilities (or scores that are transformed into probabilities). These decimal values are *thresholded* into discrete values to form the model's prediction.

# %% slideshow={"slide_type": "slide"}
# Check model predictions for the first 10 training samples

samples = x_train[:10]

# Print binary predictions ("is the digit a 5 or not?")
print(sgd_model.predict(samples))

# Print prediction probabilities
sgd_model.predict_proba(samples).round(decimals=3)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Accuracy
#
# The default performance metric for classification taks is **accuracy**.
#
# $$Accuracy = \frac{\text{Number of exact predictions}}{\text{Total number of predictions}} $$

# %%
# Define fictitious ground truth and prediction results
y_true = np.array([1, 0, 0, 1, 1, 1])
y_pred = np.array([1, 1, 0, 1, 0, 1])

# Compute accuracy: 4/6 = 2/3
acc = np.sum(y_pred == y_true) / len(y_true)
print(f"{acc:.2f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Computing training accuracy

# %% slideshow={"slide_type": "-"}
# The score function computes accuracy of the SGDClassifier
train_acc = sgd_model.score(x_train, y_train_5)
print(f"Training accuracy: {train_acc:.05f}")

# Using cross-validation to better evaluate accuracy, using 3 folds
cv_acc = cross_val_score(sgd_model, x_train, y_train_5, cv=3, scoring="accuracy")
print(f"Cross-validation accuracy: {cv_acc}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Accuracy shortcomings
#
# When the dataset is *skewed* (some classes are more frequent than others), computing accuracy is not enough to assert the model's performance.
#
# To find out why, let's imagine a dumb binary classifier that always predicts that the digit is not 5.

# %%
# Count the number of non-5 digits in the dataset
not5_count = len(y_train_5) - np.sum(y_train_5)
print(f"There are {not5_count} digits other than 5 in the training set")

dumb_model_acc = not5_count / len(x_train)
print(f"Dumb classifier accuracy: {dumb_model_acc:.05f}")


# %% [markdown] slideshow={"slide_type": "slide"}
# #### True/False positives and negatives
#
# - **True Positive (TP)**: the model _correctly_ predicts the positive class.
# - **False Positive (FP)**: the model _incorrectly_ predicts the positive class.
# - **True Negative (TN)**: the model _correctly_ predicts the negative class.
# - **False Negative (FN)**: the model _incorrectly_ predicts the negative class.
#
# $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Confusion matrix
#
# Useful representation of classification results. Row are actual classes, columns are predicted classes.
#
# [![Confusion matrix for 5s](_images/confusion_matrix.png)](https://github.com/ageron/handson-ml2)


# %% slideshow={"slide_type": "slide"}
def plot_conf_mat(model, x, y):
    """Plot the confusion matrix for a model, inputs and targets"""

    with sns.axes_style("white"):  # Temporary hide Seaborn grid lines
        _ = ConfusionMatrixDisplay.from_estimator(
            model, x, y, values_format="d", cmap=plt.colormaps.get_cmap("Blues")
        )


# Plot confusion matrix for the SGDClassifier
plot_conf_mat(sgd_model, x_train, y_train_5)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Precision and recall
#
# - **Precision**: proportion of positive identifications that were actually correct.
# - **Recall** (or *sensitivity*): proportion of actual positives that were identified correctly.
#
# $$Precision = \frac{TP}{TP + FP} = \frac{\text{True Positives}}{\text{Total Predicted Positives}}$$
#
# $$Recall = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{Total Actual Positives}}$$

# %% slideshow={"slide_type": "slide"}
# Define fictitious ground truth and prediction results
y_true = np.array([1, 0, 0, 1, 1, 1])
y_pred = np.array([1, 1, 0, 1, 0, 0])

# Compute precision and recall for both classes
for label in [0, 1]:
    TP = np.sum((y_pred == label) & (y_true == label))
    FP = np.sum((y_pred == label) & (y_true == 1 - label))
    FN = np.sum((y_pred == 1 - label) & (y_true == label))
    print(f"Class {label}: Precision {TP/(TP+FP):.02f}, Recall {TP/(TP+FN):.02f}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Example: a (flawed) tumor classifier
#
# Context: binary classification of tumors (positive means malignant). Dataset of 100 tumors, of which 9 are malignant.
#
# | Negatives | Positives |
# |-|-|
# | True Negatives: 90 | False Positives: 1 |
# | False Negatives: 8 | True Positives: 1|

# %% [markdown] slideshow={"slide_type": "fragment"}
# $$Accuracy = \frac{90+1}{100} = 91\%$$

# %% [markdown] slideshow={"slide_type": "fragment"}
# $$Precision = \frac{1}{1 + 1} = 50\%\;\;\;
# Recall = \frac{1}{1 + 8} = 11\%$$

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### The precision/recall trade-off
#
# - Improving precision typically reduces recall and vice versa ([example](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall#precision-and-recall:-a-tug-of-war)).
# - Precision matters most when the cost of false positives is high (example: spam detection).
# - Recall matters most when the cost of false negatives is high (example: tumor detection).

# %% [markdown] slideshow={"slide_type": "slide"}
# #### F1 score
#
# - Weighted average (*harmonic mean*) of precision and recall.
# - Also known as _balanced F-score_ or _F-measure_.
# - Favors classifiers that have similar precision and recall.
#
# $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

# %% slideshow={"slide_type": "slide"}
# Compute several metrics about the SGDClassifier
print(classification_report(y_train_5, sgd_model.predict(x_train)))

# %% [markdown] slideshow={"slide_type": "slide"}
# #### ROC curve and AUROC
#
# $$\text{TP Rate} = \frac{TP}{TP + FN} = Recall\;\;\;\;
# \text{FP Rate} = \frac{FP}{FP + TN}$$
#
# - ROC stands for "Receiver Operating Characteristic".
# - A ROC curve plots TPR vs. FPR at different classification thresholds.
# - AUC or more precisely AUROC ("Area Under the ROC Curve") provides an aggregate measure of performance across all possible classification thresholds.

# %% [markdown] slideshow={"slide_type": "slide"}
# [![AUROC animation](_images/auroc_animation.gif)](https://github.com/dariyasydykova/open_projects/tree/master/ROC_animation)

# %% [markdown] slideshow={"slide_type": "slide"}
# [![AUROC shape animation](_images/auroc_shape_animation.gif)](https://github.com/dariyasydykova/open_projects/tree/master/ROC_animation)

# %% slideshow={"slide_type": "slide"}
# Plot ROC curve for the SGDClassifier
sgd_disp = RocCurveDisplay.from_estimator(sgd_model, x_train, y_train_5)
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Multiclass classification

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Choosing a loss function
#
# The log loss extends naturally to the multiclass case. It is also called **Negative Log-Likelihood** or **Cross Entropy**, and is also implemented in the scikit-learn [log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) function.
#
# $$\mathcal{L}_{\mathrm{CE}}(\pmb{\omega}) = -\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K y^{(i)}_k \log_e(y'^{(i)}_k))$$
#
# - $\pmb{y^{(i)}} \in \{0,1\}^K$: binary vector of $K$ elements.
# - $y^{(i)}_k \in \{0,1\}$: expected value for the $k$th label of the $i$th sample. $y^{(i)}_k = 1$ iff the $i$th sample has label $k \in [1,K]$.
# - $y'^{(i)}_k \in [0,1]$: model output for the $k$th label of the $i$th sample, i.e. probability that the $i$th sample has label $k$.


# %% slideshow={"slide_type": "slide"}
# Compute cross entropy losses for pseudo-predictions

# 2 samples with 3 possibles labels. Sample 1 has label 2, sample 2 has label 3
y_true = [[0, 1, 0], [0, 0, 1]]

# Probability distribution vector
# 95% proba that sample 1 has label 2, 70% proba that sample 2 has label 3
y_pred = [[0.05, 0.95, 0], [0.1, 0.2, 0.7]]

# Compute cross entropy loss
ce = log_loss(y_true, y_pred)
print(f"Cross entropy loss: {ce:.05f}")

# Compare theorical and computed loss values
np.testing.assert_almost_equal(-(np.log(0.95) + np.log(0.7)) / 2, ce)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Training a multiclass classifier

# %% slideshow={"slide_type": "-"}
# Using all digits as training results
y_train = train_targets
y_test = test_targets

# Training another SGD classifier to recognize all digits
multi_sgd_model = SGDClassifier(loss="log_loss")
multi_sgd_model.fit(x_train, y_train)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Assessing performance

# %%
# Since dataset is not class imbalanced anymore, accuracy is now a reliable metric
print(f"Training accuracy: {multi_sgd_model.score(x_train, y_train):.05f}")
print(f"Test accuracy: {multi_sgd_model.score(x_test, y_test):.05f}")

# %% slideshow={"slide_type": "slide"}
# Plot confusion matrix for the multiclass SGD classifier
plot_conf_mat(multi_sgd_model, x_train, y_train)

# %% slideshow={"slide_type": "slide"}
# Compute performance metrics about the multiclass SGD classifier
print(classification_report(y_train, multi_sgd_model.predict(x_train)))

# %%
