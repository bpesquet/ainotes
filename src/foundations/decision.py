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
# # Decision-making
#
# ```{warning}
# This chapter is under construction.
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Terminology

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Decision
#
# A **decision** is "a deliberative process that results in the commitment to a categorical proposition." {cite}`goldNeuralBasisDecision2007`
#
# People make thousands of (big and small) decisions everyday. A few examples:
# - Choosing what pair of socks to wear.
# - Deciding on a TV series to watch.
# - Choosing one's next car/bike.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Decision-making
#
# Decision-making designates the cognitive process resulting in a decision.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The sequential nature of decision-making
#
# Humans and animals make their decisions after a **deliberation** phase.
#
# Many decisions are based on information that unfolds over time (example: cues for solving an homicide).
#
# Even if all informative data is immediately available (example: a chess position), it has to be treated **sequentially** by our nervous system, reflecting its inability to process information simultaneously.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### From stimulus to response
#
# **Decision-making** can be formalized as the mapping from a stimulus to a response.
#
# Time between stimulus and response execution is called **Reaction Time** or **Response Time** (RT) {cite}`forstmannSequentialSamplingModels2016`, {cite}`myersPracticalIntroductionUsing2022`.
#
# $$RT = T_e+T_d+T_r$$
#
# $T_{er}= T_e+T_r$ is called **non-decision time**.
#
# ![Response Time](_images/response_time.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Speed/accuracy tradeoff
#
# All decisions are made under time pressure {cite}`forstmannSequentialSamplingModels2016`. The balance between response time and accuracy is called the **speed/accuracy trade-off**.
#
# It is at least partially under conscious control: decision-makers can decide to make faster decisions at the expense of an higher error rate, or slower, more accurate decisions {cite}`ratcliffDiffusionDecisionModel2016`. This complicates the interpretation of behavioral data.
#
# Ideally, what is needed is a way to evaluate data that considers not only accuracy and speed, but the interaction between them {cite}`myersPracticalIntroductionUsing2022`.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The cognitive processes of decision-making
#
# ![Cognitive processes of decision-making](_images/decision-making_processes.png)
#
# {cite}`bogaczOptimalDecisionmakingTheories2007`

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Modeling speeded decision-making

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Context
#
# One approach to understanding decision-making is through **computational models**.
#
# Several models have been developed to account for the speed/accuracy trade-off and explain how people and animals make decisions under time pressure.
#
# Historically, most research on the dynamics of decision-making has been focused on simple, repeatable problems involving fast binary-choice decisions with one correct answer.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Task examples
#
# - Lexical decision tasks: pressing one key if the stimulus is a word or another if it is a non-word.
# - [Stroop tasks](https://en.wikipedia.org/wiki/Stroop_effect).
# - Saccadic flanker tasks: moving the eye in the direction indicated by a central stimulus, ignoring the directionality of flanker stimuli.
# - *Random Dot Kinematogram* (RDK): judging whether a subset of dots move to the left or to the right.
#
# ![RDK](_images/rdk.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Measures of interest
#
# - Response times (RTs) for correct responses and for error responses.
# - Distributions of RTs across trials.
# - Proportion of correct responses (accuracy).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Sequential sampling
#
# A popular class of models assumes that, on each trial, the decision maker accumulates noisy samples of information from the environment until a **threshold** of evidence is reached.
#
# Such accumulation-to-threshold models are known as **sequential sampling models**, a.k.a. **evidence accumulation models**.
#
# Different approaches to sequential sampling coexist. A key distinction is the number of **accumulators** (structures for gathering evidence in favor of one response) and whether they are independent from one another.
#
# An accumulator is also called a **Decision Variable** (DV).

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Example: sequential sampling for RDK
#
# ![Sequential sampling for RDK](_images/eam_rdk.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The sequential sampling model family
#
# ![Sequential Sampling model landscape](_images/sequential_sampling_models.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Accumulator models
#
# These models, also known as **race models**,  have independent accumulators (one per possible choice) and an **absolute** evidence response rule (two fixed thresholds, once for each accumulator). The process stops once one of the accumulators reaches its threshold.
#
# ![Illustration of an accumulator model](_images/race_model.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Random walk models
#
# These models use a **relative** evidence rule: a response is initiated as soon as the difference in accumulated evidence reaches a predefined threshold, also called a **criterion**.
#
# ![Illustration of a random walk model](_images/random_walk.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Problem example: the gambler's ruin
#
# Two gamblers A and B have a (possibly different) starting capital and play a series of independant games against each other. Every game has a fixed chance $p$ of being won by gambler A. After each game, the winner obtains one unit of the other player’s capital. The process continues until one of the gamblers is bankrupt.
#
# This problem can be modelised as a random walk, with $p$ representing the **drift** of the process. Depending on the value of $p$, the process will drift towards one of the thresholds (gambler B’s bankruptcy if $p>0.5$).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Mathematical modeling
#
# - $K$: number of possible states of the world. For binary choices, $K=2$.
# - $h_k, k \in [1,K]$: hypothesis (state). Examples: dot movement type, presence of a stimulus, etc.
# - $H_k, k \in [1,K]$: choice associated with hypothesis $h_k$.
# - $P(h_k)$: probability that $h_k$ is true before obtaining any evidence about it.
# - $n$: number of evidences received.
# - $e_i, i \in [1,n]$: evidence (noisy information) and guiding commitment to a particular hypothesis $h_k$.
# - $P(e_i|h_k)$: likelihood (values that $e_i$ can attain when $h_k$ is true).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Sequential Probability Ratio Test
#
# Particular form of **sequential analysis** for binary decisions ($K=2$) where the DV is constructed from multiple, independent pieces of evidence $e_1, e_2, \dots, e_n$ as the logarithm of the likelihood ratio between hypotheses $h_1$ and $h_2$.
#
# $$DV_n = \sum_{i=1}^n log \frac{P(e_i|h_1)}{P(e_i|h_2)} = \sum_{i=1}^n w_i$$
#
# - $w_i, i \in [1,n]$: weight of the $i$th evidence.
#
# The DV is updated with new pieces of evidence until reaching a criterion.
#
# SPRT is the most efficient statistical test for deciding between two hypotheses on this kind of problem: it achieves a desired error rate with the smallest number of samples, on average {cite}`waldOptimumCharacterSequential1948`.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Problem example: trick or fair coin
#
# Two coins are placed in a bag: one is fair (50% chance of obtaining heads or tails when tossing it), the other not (60/40). One of the coins is drawn from the bag: is it a trick ($h_1$) or a fair ($h_2$) coin? How many tosses are needed for this decision?
#
# $$\forall i \in[1,n], w_i=
#     \begin{cases}
#       \log \frac{P(e_{heads}|h_1)}{P(e_{heads}|h_2)} = \log \frac{0.6}{0.5} = 0.182 & \text{if toss gives "heads"} \\
#       \log \frac{P(e_{tails}|h_1)}{P(e_{tails}|h_2)} = \log \frac{0.4}{0.5} = -0.223 & \text{if toss gives "tails"}
#     \end{cases}$$
#
# $$\text{If}\ DV_n \ge \frac{1-\alpha}{\alpha}, \text{answer "trick".}\ \text{If}\ DV_n \le \frac{\alpha}{1-\alpha}, \text{answer "fair".}$$
#
# With $\alpha = P(H_2|h_1) = P(H_1|h_2)$ the probability of misidentifying a coin.
#
# For $\alpha=0.05$, a decision happens when $|DV_n| \ge \log(19)$.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Diffusion models
#
# When the evidence is continously sampled from a distribution in infinitesimal time steps, the process is termed **diffusion** with drift $\mu$. When $\mu$ is constant, this process is known as [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion) or Wiener diffusion process {cite}`goldNeuralBasisDecision2007`.
#
# ![Diffusion model](_images/diffusion_model.png)
#
# > The diffusion models for decision-making are not to be confused with the [diffusion models of Machine Learning](https://en.wikipedia.org/wiki/Diffusion_model).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The Diffusion Decision Model
#
# This model, also called **Drift Diffusion Model (DDM)**, is a sequential sampling model for binary choices in continuous environments {cite}`ratcliffDiffusionDecisionModel2008`.
#
# Originally designed in the 1970's {cite}`ratcliff1978theory`, it has recently experienced a surge in popularity. A growing body of literature is using the DDM to elucidate the cognitive processes of decision-making.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Parameters
#
# |Name|Description|Typical range|Implements|
# |-|-|-|-|
# |$a$|Boundary separation|$[0.5,2]$ (in arbitrary units)|Speed/accuracy trade-off|
# |$z$|Starting point|$[0,1]$ (as proportion of $a$)|Response bias|
# |$v$|Drift rate|$[-5,5]$|Speed of evidence accumulation processing|
# |$T_{er}$|Non-decision time|$[0.1,0.5]$ (in seconds)|Neurological processes for stimulus encoding and motor response|
#
# ![DDM](_images/ddm.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Advantages
#
# - Experimental validation has shown that:
#   - DDM parameters do capture recognizable, and at least partly separable, cognitive processes.
#   - DDM provides an explanation for some of the dynamics of neuronal activity in brains.
# - Several software packages like [HDDM](https://hddm.readthedocs.io/en/latest/) facilitate fitting the model to experimental data, or generating simulated data.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Applications
#
# ##### Behavioral analysis
#
# - **Aging**: DDM-based studies showed that older adults had slower non-decision times and set wider boundaries, but their drift rates were not always lower than those of young adults.
# - **IQ**: DDM-based analyses showed showed that drift rate varied with IQ, but boundary separation and nondecision time did not.
# - Other studies showed that **sleep deprivation** and **alcohol consumption** lower drift rate, but have either small or no effect on boundary separation and non-decision time.
# - ...

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Low-level neuroscience
#
# Studies {cite}`Gold2001-xr` uses DDM as inspiration to interpret neuron firing rates in monkeys as evidence accumulation until a threshold is reached.
#
# ![Firing rate in monkeys](_images/ddm_firing_rate.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### High-level neuroscience
#
# Studies correlate parameter estimates from DDM models to the blood-oxygen-level dependent signal obtained from fMRI experiments in perceptual decision-making.
#
# ![Brain activity & DDM](_images/ddm_brain_activity.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Extension to dynamic thresholds
#
# Collapsing-bound models translate the fact that in some cases, decisions are based on less and less evidence as time passes.
#
# ![DDM extension:dynamic thresholds](_images/ddm_dyn_thresholds.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Relationships between models
#
# Cortical models ({cite}`Shadlen2001-lo`, {cite}`Usher2001-di`, {cite}`Wang2002-vn`) can be reduced to the diffusion model for parameter values that optimize their performance {cite}`bogaczOptimalDecisionmakingTheories2007`.
#
# ![DDM & cortical models](_images/ddm_cortical_models.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Learning and decision-making

# %%
