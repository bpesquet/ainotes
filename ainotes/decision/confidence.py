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
# # Confidence in decision-making
#
# ```{warning}
# This chapter is under construction.
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Terminology

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Confidence
#
# In general terms, **confidence** is the belief or conviction that a hypothesis or prediction is correct, that an outcome will be favorable, or that a chosen course of action is the best or most effective.
#
# In decision-making, confidence can be more precisely defined as **the subjective estimate of decision quality** {cite}`brusSourcesConfidenceValuebased2021`.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Trust
#
# **Trust** is a social construct: the belief that someone or something will behave or perform as expected. It implies a relationship between a *trustor* and a *trustee*.
#
# **Self-confidence** is trust in one's abilities.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Uncertainty
#
# Generally speaking, **uncertainty** (or incertitude) characterizes situations involving imperfect or unknown information.
#
# In decision-making, it refers to **the variability in the representation of information before a decision is taken** {cite}`mamassian2020`.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Belief

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Bias

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Sensitivity

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Error monitoring
#
# In decision-making, **error monitoring (EM)** is the process by which one is able to detect his/her errors as soon as a response has been made {cite}`yeungMetacognitionHumanDecisionmaking2012`.
#
# EM allows adaptation of behavior both in the short and longer terms through gradual learning of actions' outcomes.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Cognitive control

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Metacognition
#
# Confidence judgments and error monitoring are two related aspects of **metacognition**, the self-monitoring and self-control of one's own cognition (sometimes called *high order thinking*).
#
# [![Metacognition](_images/metacognition.jpg)](https://commons.wikimedia.org/wiki/File:Metacognition.jpg)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Usefulness of confidence in decision-making

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Modeling decision confidence

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Signal Detection Theory
#
# Framework for analyzing decision making in the presence of uncertainty.
#
# Originally developped by radar researchers in the mid-20th century, it has applications in many fields (psychology, diagnostics, quality control, etc).

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Sensitivity and specificity
#
# **Sensitivity** quantifies how well a model can identify true positives. **Specificity** quantifies how well a model can identify true negatives. Equivalent to the recall metric, these definitions are often used in [medecine and statistics](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
#
# $$\text{Sensitivity} = \frac{TP}{TP + FN} = \text{True Positive Rate} = \text{Recall}_{positive}$$
#
# $$\text{Specificity} = \frac{TN}{TN + FP} = \text{True Negative Rate} = \text{Recall}_{negative}$$

# %% [markdown] slideshow={"slide_type": "slide"}
# Prediction outcomes can be interpreted as probability density functions, in order to represent results graphically.
#
# [![Sensitivity and specificity](_images/Specificity_vs_Sensitivity_Graph.png)](https://en.wikipedia.org/wiki/Sensitivity_and_specificity#/media/File:Specificity_vs_Sensitivity_Graph.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### ROC curve and AUROC
#
# $$\text{False Positive Rate} = \frac{FP}{TN+FP} = 1 - TNR = 1 -\text{Specificity}$$
#
# - ROC stands for "Receiver Operating Characteristic".
# - A ROC curve plots sensitivity vs. (1 - specificity), or TPR vs. FPR, for each possible classification threshold.
# - AUC, or more precisely AUROC ("Area Under the ROC Curve"), provides an aggregate measure of performance across all possible classification thresholds.

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Impact of threshold choice
#
# [![AUROC animation](_images/auroc_animation.gif)](https://github.com/dariyasydykova/open_projects/tree/master/ROC_animation)

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Impact of model's separative power
#
# [![AUROC shape animation](_images/auroc_shape_animation.gif)](https://github.com/dariyasydykova/open_projects/tree/master/ROC_animation)

# %% [markdown]
# #### Discriminability index

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Measuring confidence
#
# Two dominant methodologies:
#
# - Confidence ratings: after a decision, evaluate its correctness.
# - Confidence forced choice: after two decisions, choose which one is more likely to be correct.
#   - Disregards confidence biases to focus on confidence sensitivity.

# %%
