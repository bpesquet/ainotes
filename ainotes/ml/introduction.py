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
# # Introduction to Machine Learning

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Learning objectives
#
# - Know what Machine Learning and Deep Learning are about.
# - Understand the main categories of ML systems.
# - Discover some of the many existing ML algorithms.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "-"}
import platform
from IPython.display import YouTubeVideo

print(f"Python version: {platform.python_version()}")


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Whats is Machine Learning?


# %% [markdown] slideshow={"slide_type": "slide"}
# ### The first definition of Machine Learning
#
# > "The field of study that gives computers the ability to learn without being explicitly programmed." (Arthur Samuel, 1959).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Machine Learning in a nutshell
#
# Set of techniques for giving machines the ability to to find **patterns** and extract **rules** from data, in order to:
#
# * **Identify** or **classify** elements.
# * Detect **tendencies**.
# * Make **predictions**.
#
# As more data is fed into the system, results get better: performance improves with experience.
#
# a.k.a. **Statistical Learning**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### A new paradigm...

# %% [markdown] slideshow={"slide_type": "fragment"}
# ![Programming paradigm](_images/programming_paradigm.png)

# %% [markdown] slideshow={"slide_type": "fragment"}
# ![Training paradigm](_images/training_paradigm.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### ... Or merely a bag of tricks?
#
# [![ML on XKCD](_images/ml_xkcd.png)](https://xkcd.com/1838/)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## The Machine Learning landscape

# %% [markdown] slideshow={"slide_type": "slide"}
# ### AI, Machine Learning and Deep Learning
#
# ![AI/ML/DL Venn diagram](_images/ai_ml_dl.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Typology of ML systems
#
# ML systems are traditionally classified in three categories, according to the amount and type of human supervision during training. [Hybrid approaches](https://hackernoon.com/self-supervised-learning-gets-us-closer-to-autonomous-learning-be77e6c86b5a) exist.
#
# - **Supervised Learning**: expected results (called *labels* or *tags*) are given to the system along with training data.
# - **Unsupervised Learning**: training data comes without the expected results. The system must discover some structure in the data by itself.
# - **Reinforcement Learning**: without being given an explicit goal, the system's decisions produce a **reward** it tries to maximize.

# %% [markdown] slideshow={"slide_type": "slide"}
# ![ML category tree](_images/ml_tree.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Regression
#
# The system predicts **continuous** values. Examples: temperature forecasting, asset price prediction...
#
# ![Regression example](_images/ml_regression.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Classification
#
# The system predicts **discrete** values: input is **categorized**.
#
# ![Classification example](_images/ml_classification.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Classification types
#
# - **Binary**: only two possibles classes. Examples: cat/not a cat, spam/legit mail, benign/malignant tumor.
# - **Multiclass**: several mutually exclusive classes. Example: handwritten digit recognition.
# - **Multilabel**: several non-mutually exclusive classes. Example: face recognition.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Clustering
#
# Data is partitioned into groups.
#
# ![ML clustering example](_images/ml_clustering.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Anomaly Detection
#
# The system is able to detect abnomal samples (*outliers*).
#
# ![ML anomaly detection example](_images/ml_anomaly_detection.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Game AI

# %% slideshow={"slide_type": "-"} tags=["hide-input"]
YouTubeVideo("TmPfTpjtdgg")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## How do machines learn, actually?

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Algorithm #1: K-Nearest Neighbors
#
# Prediction is based on the `k` nearest neighbors of a data sample.
#
# [![K-NN](_images/knn.png)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Algorithm #2: Decision Trees
#
# Build a tree-like structure based on a series of discovered questions on the data.
#
# ![Decision Tree for Iris dataset](_images/dt_iris.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Algorithm #3: Neural Networks
#
# Layers of loosely neuron-inpired computation units that can approximate any continuous function.
#
# ![Neuron output](_images/neuron_output.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ![Dog or Cat?](_images/neural_net.gif)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## The advent of Deep Learning

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The Deep Learning tsunami
#
# DL is a subfield of Machine Learning consisting of multilayered neural networks trained on vast amounts of data.
#
# [![AlexNet'12 (simplified)](_images/alexnet.png)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
#
# Since 2010, DL-based approaches outperformed previous state-of-the-art techniques in many fields (language translation, image and scene recognition, and [much more](https://huggingface.co/spaces/akhaliq/AnimeGANv2)).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Reasons for success
#
# - Explosion of available data.
# - Huge progress in computing power.
# - Refinement of many existing algorithms.
# - Availability of sophisticated tools for building ML-powered systems.
#
# ![TF, Keras and PyTorch logos](_images/tf_keras_pytorch.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ![Big data universe](_images/big_data_universe.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ![Computer power sheet](_images/infographic2-intel-past-present.gif)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### From labs to everyday life in 25 years
#
# [![LeCun - LeNet](_images/lecun_lenet.gif)](http://yann.lecun.com/exdb/lenet/)
#
# [![Facial recognition in Chinese elementary school](_images/china_school_facial_reco.gif)](https://twitter.com/mbrennanchina/status/1203687857849716736)

# %%
