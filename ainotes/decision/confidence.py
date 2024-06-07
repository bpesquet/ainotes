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
# ## Environment setup

# %% slideshow={"slide_type": "slide"}
import platform

import scipy

# Print environment info
print(f"Python version: {platform.python_version()}")
print(f"SciPy version: {scipy.__version__}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Terminology

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Confidence: general definition
#
# Everyone knows intuitively what confidence is about, yet it is seldom defined explicitely.
#
# In the broadest sense, **confidence quantifies a degree of belief in something or someone** {cite}`meynielConfidenceBayesianProbability2015`.
#
# It is fundamentally linked to its object: a thought, a choice, an external actor, etc.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Belief
#
# A belief is **a feeling of certainty about a proposition**. It is a subjective, conscious experience.
#
# Regarding perceptions, our belief usually matches our perceptual experience, but not always {cite}`mamassianConfidenceForcedChoiceOther2020`.
#
# ![Belief-perception gap](_images/belief_perception_gap.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Uncertainty
#
# Generally speaking, uncertainty (or incertitude) characterizes situations involving **imperfect or unknown information**.
#
# In decision-making, uncertainty refers to **the variability in the representation of information before a decision is taken** {cite}`mamassianConfidenceForcedChoiceOther2020`.
#
# To perform well, the brain needs to be effective at dealing with many uncertainties, some of them external (changes in world state or sensorimotor variability), others internal (cognitive variables, timing or abstract states). Uncertainty is inherent to all stages of neural computation.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Confidence: updated definition
#
# In decision-making, confidence can be seen as **the subjective estimate of decision quality** {cite}`brusSourcesConfidenceValuebased2021`.
#
# More formally, it can be defined as **the probability that a choice is correct given the evidence** {cite}`pougetConfidenceCertaintyDistinct2016`.
#
# Confidence is a form of certainty. A key difference is that contrary to confidence, (un)certainties are *decision independant*. **Confidence quantifies the degree of certainty associated to a decision**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Trust
#
# Trust is a social construct: **the belief that someone or something will behave or perform as expected**. It implies a relationship between a *trustor* and a *trustee*.
#
# **Self-confidence** is trust in one's abilities.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Error monitoring
#
# In decision-making, error monitoring (EM) is **the process by which one is able to detect his/her errors as soon as a response has been made** {cite}`yeungMetacognitionHumanDecisionmaking2012`.
#
# EM allows adaptation of behavior both in the short and longer terms through gradual learning of actions' outcomes.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Metacognition
#
# Confidence judgments and error monitoring are two related aspects of metacognition (sometimes called *higher order thinking*).
#
# First described in {cite}`flavellMetacognitionCognitiveMonitoring1979`, metacognition can be defined as **the ability to consider, understand and regulate one's cognitive processes**. It is a key skill to adapt to complex problems and changing environments.
#
# Metacognition is classicaly divided into two subprocesses: **monitoring** and **control**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ![Metacognition diagram](_images/metacognition_diagram.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Metaperception
#
# ![Metaperception](_images/metaperception.png)
#
# {cite}`mamassianConfidenceForcedChoiceOther2020`

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Cognitive control
#
# Cognitive control refers to **the intentional selection of thoughts, emotions, and behaviors based on current task demands and social context, and the concomitant suppression of inappropriate habitual actions** {cite}`millerIntegrativeTheoryPrefrontal2001`.
#
# In simpler terms, cognitive control allows adapting our behaviour on-the-fly to improve performance.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Measuring confidence

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Measures of interest

# %% [markdown] slideshow={"slide_type": "-"}
# #### Sensitivity
#
# Confidence/metacognitive/type 2 sensitivity is **the capacity to distinguish correct from incorrect decisions**.
#
# Being confident when taking correct decisions and less confident otherwise demonstrates a high degree of sensitivity.
#
# Sensitivity is often affected by task performance itself: an individual will appear to have greater sensitivity on an easy task compared to a hard task {cite}`flemingHowMeasureMetacognition2014`.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Bias
#
# Confidence/metacognitive/type 2 bias is **a difference in subjective confidence despite constant task performance**.
#
# Under- and over-confidence are examples of biases.
#
# ![Sensitivity Vs bias](_images/sensitivity_vs_bias.png)
#
# > Real confidence distributions are unlikely to be Gaussian.
#
# {cite}`flemingHowMeasureMetacognition2014`

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Efficiency
#
# Confidence/metacognitive efficiency is **the level of sensitivity given a certain level of task performance**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Experimental tasks
#
# Their setup is similar to those used to study [decision-making](introduction.ipynb). The major difference is that after taking a decision (a *type 1* task), subjects express their confidence (a *type 2* task).
#
# Type 1 task example: is the [Gabor patch](http://neuroanatody.com/2016/05/whats-in-a-gabor-patch/) tilted to the left or to the right?
#
# ![Gabor patch](_images/gabor_patch.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Flow of information for a perceptual task
#
# ![Flow of information for a perceptual task](_images/confidence_information_flow.png)
#
# {cite}`mamassianConfidenceForcedChoiceOther2020`

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Measurement methods

# %% [markdown] slideshow={"slide_type": "-"}
# #### Confidence ratings
#
# ![Confidence ratings](_images/confidence_ratings.png)
#
# After a decision, the subject is asked to evaluate its correctness, using a dedicated scale.
#
# Simple and frequently used, this method has several drawbacks: intersubject variability regarding bias and scale usage, and possible confusions between type 1 and type 2 judgments {cite}`mamassianConfidenceForcedChoiceOther2020`.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Post-decision wagering
#
# After a decision, subjects are asked to gamble on whether their response was correct. If the decision is correct, the wager amount is kept {cite}`flemingNeuralBasisMetacognitive2012`.
#
# The amount of the bet is assumed to reflect a subject’s confidence in his or her decision.
#
#

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Opt-out paradigm
#
# ![Confidence opt-out](_images/confidence_opt_out.png)
#
# In most but not all the trials, the subject has the option to decline the decision task and receive a smaller reward.
#
# This paradigm is well suited to experiments with animals, which cannot explicitely report their confidence.
#
# One challenge is to avoid confounding it with a three-alternative forced choice {cite}`kepecsComputationalFrameworkStudy2012`.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Confidence forced choice
#
# ![Confidence forced choice](_images/confidence_forced_choice.png)
#
# After two decisions, the subject has to choose which one is more likely to be correct.
#
# One benefit of this paradigm is that it disregards confidence biases to focus on sensitivity {cite}`mamassianConfidenceForcedChoiceOther2020`.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Computing confidence

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Context
#
# Let's consider a simple two-alternative forced choice decision task.
#
# Assuming that post-decisional confidence was measured on a binary scale (high/low), we can count the number of confidence ratings assigned to each judgment in the following type 2 table.
#
# |Type 1 decision|Low confidence|High confidence|
# |-|-|-|
# |Incorrect|True Negatives ($TN_2$)|False Positives ($FP_2$)|
# |Correct|False Negatives ($FN_2$)|True Positives ($TP_2$)|

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Phi/Pearson correlation
#
# The simplest measure of confidence sensitivity is the $\phi$ correlation (a.k.a. Pearson $r$ correlation for binary variables) between accuracy and confidence measurements.
#
# > This metric is equivalent to the Matthews Correlation Coefficient (MCC) in the context of Machine Learning.

# %% slideshow={"slide_type": "slide"}
# Binary results for each decision (type 1 task)
# 1 = correct answer, 0 = wrong answer
results = [1, 0, 0, 1, 1, 0]

# Binary confidence for each decision (type 2 task)
# 1 = high confidence, 0 = low confidence
confidence = [1, 1, 0, 0, 1, 0]

r = scipy.stats.pearsonr(results, confidence)
print(f"Correlation: {r}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Signal Detection Theory
#
# Framework for analyzing decision making in the presence of uncertainty.
#
# Originally developped in the mid-20th century by radar researchers to assess how faithfully an operator is able to separate signal from noise, it has applications in many fields (psychology, diagnostics, quality control, etc).

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
# ##### Impact of criterion choice
#
# [![AUROC animation](_images/auroc_animation.gif)](https://github.com/dariyasydykova/open_projects/tree/master/ROC_animation)

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Impact of task difficulty
#
# [![AUROC shape animation](_images/auroc_shape_animation.gif)](https://github.com/dariyasydykova/open_projects/tree/master/ROC_animation)

# %% [markdown]
# #### Discriminability index

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Exploiting confidence

# %%
