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
# # Principles of supervised learning

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Terminology

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Components of a supervised ML system
#
# - Some **data** to learn from.
# - A **model** to transform data into results.
# - A **loss function** to quantify how well (or badly) the model is doing.
# - An **optimization algorithm** to update the model according to the loss function.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Features
#
# A **feature** is an attribute (property) of the data given to the model: the number of rooms in a house, the color of a pixel in an image, the presence of a specific word in a text, etc. Most of the time, they come under numerical form.
#
# A simple ML project might use a single feature, while more sophisticated ones could use millions of them.
#
# They are denoted using the $x$ variable.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Label
#
# A **label** (or **class** in the context of classification), is a result the model is trying to predict: the future price of an asset, the nature of the animal shown in a picture, the presence or absence of a face, etc.
#
# They are denoted using the $y$ variable.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Samples
#
# An **sample**, also called **example**, is a particular instance of data: an individual email, an image, etc.
#
# A **labeled sample** includes both its feature(s) and the associated label(s) to predict. An **unlabeled sample** includes only feature(s).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Inputs
#
# **Inputs** correspond to all features for one sample of the dataset.
#
# They are denoted using the $\pmb{x}$ variable (notice the boldface to indicate that it is a vector).
#
# $$\pmb{x}^{(i)} = \begin{pmatrix}
#        \ x^{(i)}_1 \\
#        \ x^{(i)}_2 \\
#        \ \vdots \\
#        \ x^{(i)}_n
#      \end{pmatrix}$$
#
# - $m$: number of samples in the dataset.
# - $n$: number of features for one sample.
# - $\pmb{x}^{(i)}, i \in [1,m]$: vector of $n$ features.
# - $x^{(i)}_j, j \in [1,n]$: value of the $j$th feature for the $i$th data sample..

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Targets
#
# **Targets** are the expected results (labels) associated to a data sample, often called the *ground truth*. They are denoted using the $\pmb{y}$ variable.
#
# Some ML models have to predict more than one value for each sample (for example, in multiclass classification).
#
# $$\pmb{y}^{(i)} = \begin{pmatrix}
#        \ y^{(i)}_1 \\
#        \ y^{(i)}_2 \\
#        \ \vdots \\
#        \ y^{(i)}_K
#      \end{pmatrix}$$
#
# - $K$: number of labels associated to a data sample.
# - $\pmb{y}^{(i)}, i \in [1,m]$: vector of $K$ labels.
# - $y^{(i)}_k, k \in [1,K]$: value of the $k$th label for the $i$th sample.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Inputs matrix
#
# Many ML models expect their inputs to come under the form of a $m \times n$ matrix, often called **design matrix** and denoted $\pmb{X}$.
#
# $$\pmb{X} = \begin{bmatrix}
#        \ \pmb{x}^{(1)T} \\
#        \ \pmb{x}^{(2)T} \\
#        \ \vdots \\
#        \ \pmb{x}^{(m)T} \\
#      \end{bmatrix} =
# \begin{bmatrix}
#        \ x^{(1)}_1 & x^{(1)}_2 & \cdots & x^{(1)}_n \\
#        \ x^{(2)}_1 & x^{(2)}_2 & \cdots & x^{(2)}_n \\
#        \ \vdots & \vdots & \ddots & \vdots \\
#        \ x^{(m)}_1 & x^{(m)}_2 & \cdots & x^{(m)}_n
#      \end{bmatrix}$$

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Targets matrix
#
# Accordingly, expected results are often stored in a $m \times K$ matrix denoted $\pmb{Y}$.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Model
#
# The representation learnt from data during training is called a **model**. It defines the relationship between features and labels.
#
# Most (but not all) ML systems are model-based.
#
# [![Extract from the book Hands-on Machine Learning with Scikit-Learn & TensorFlow by A. Géron](_images/instance_model_learning.png)](https://github.com/ageron/handson-ml2)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The two phases of a model's life
#
# - **Training**: using labeled samples, the model learns to find a relationship between features and labels.
# - **Inference**: the trained model is used to make predictions on unlabeled samples (new data unseen during training).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Model parameters and hyperparameters
#
# **Parameters**, sometimes called **weights**, are the internal values that affect the computed output of a model. During the training phase, they are algorithmically adjusted for optimal performance w.r.t the loss function. The set of parameters for a model is denoted $\pmb{\omega}$ or $\pmb{\theta}$.
#
# They are not to be confused with **hyperparameters**, which are configuration properties that constrain the model: the maximum depth of a decision tree, the number of layers in a neural networks, etc. Hyperparameters are statically defined before training by the user or by a dedicated tool.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Hypothesis function
#
# Mathamatically speaking, a model is a function of the inputs that depends on its parameters and computes results (which will be compared to targets during the training process).
#
# This function, called the **hypothesis function**, is denoted $h_{\pmb{\omega}}$. Its output (predicted result) is denoted  $\pmb{y'}$ or $\hat{\pmb{y}}$.
#
# $$\pmb{y'}^{(i)} = h_{\pmb{\omega}}(\pmb{x}^{(i)})$$
#
# - $\pmb{y'}^{(i)}, i \in [1,m]$: predicted output for the $i$th sample.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Loss function
#
# The **loss function**, also called **cost function** or **objective function**, quantifies the difference, often called **error**, between targets (expected results) and actual results computed by the model. Its value at any given time is called the **loss value**, or simply **loss**.
#
# By convention, loss functions are usually defined so that lower is better, hence their name. If the model's prediction is perfect, the loss value is zero.
#
# The loss function is denoted $\mathcal{L}$ or $\mathcal{J}$.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Loss function example
#
# The choice of the loss function depends on the problem type.
#
# For regression tasks, a popular choice is the **Mean Squared Error** or squared L2 norm.
#
# $$\mathcal{L}_{MSE} = \frac{1}{m}\sum_{i=1}^m (\pmb{y'}^{(i)} - \pmb{y}^{(i)})^2$$

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Optimization algorithm
#
# Used only during the training phase, it aims at finding the set of model parameters (denoted $\pmb{\omega^*}$ or $\pmb{\theta^*}$) that minimizes the loss value.
#
# Depending on the task and the model type, several algorithms of various complexity exist.
#
# [![Untrained Vs trained model](_images/LossSideBySide.png)](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Workflow

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Main steps of an ML project
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

# %%
