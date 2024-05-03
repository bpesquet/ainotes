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
# # Probability & statistics
#
# > "Probability deals with knowing the model of our world and use it for deductions. Statistics deal with trying to identify this model based on observation of reality."
#
# ```{warning}
# This chapter is under construction.
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "slide"}
import platform

import torch

# Print environment info
print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Fundamentals

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Random experiment (a.k.a. trial)
#
# Experiment implying some randomness. Knowing the experimental conditions is not enough to predict its outcome. Opposite of *deterministic experiment*.
#
# Example: throwing a 6-sided dice.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Sample space
#
# Set of all possible outcomes for the random experiment, also called *universe* of the random experiment. Denoted by $\Omega$ or $S$. Can either be finite or infinitely large.
#
# Example: $\Omega = \{1,2,3,4,5,6\}$.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Event
#
# An **event** $A$ is a subset of the sample space: $A \subseteq \Omega$.
#
# $\Omega$ is the certain event. $\emptyset$ is the impossible event.
#
# An event is a set of **elementary events**. An elementary event $\omega \in \Omega$ is an observed outcome of the random experiment. If $\omega \in A$, then event $A$ has occurred.
#
# Example: $A \equiv \text{"obtaining an odd number"}$. $A= \{1,3,5\}$.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Relationships between events
#
# * Inclusion $A \subset B$. If $A$ occurs, $B$ occurs too.
# * Equality $A = B$. All elements are identical for both events.
# * Incompatibility (no shared elements). The events cannot occur simultaneously.
#
# Example: $A \equiv \text{"obtaining an odd number"}$ and $B \equiv \text{"obtaining an even number"}$ are incompatible.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Composition of events
#
# * Negation $A^\star = \Omega \backslash A$ ($\Omega$ minus $A$). Occurs if and only if $A$ does not occur.
# * Conjunction $A \cap B$. Occurs if and only if $A$ and $B$ occur simultaneously.
# * Union $A \cup B$. Occurs if either $A$ or $B$ occurs.
# * Generalization to $n$ events:
#   * $\bigcap_{i=1}^{n} A_i =  A_1 \cap A_2 \cap \cdots \cap A_n$ (All $A_i$ events occured simutaneously)
#   * $\bigcup_{i=1}^{n} A_i =  A_1 \cup A_2 \cup \cdots \cup A_n$ (At least one of the $A_i$ events occurred)
#   * $(\bigcap_{i=1}^{n} A_i) \subseteq (\bigcup_{i=1}^{n} A_i)$
#
# Example: $A \equiv \text{"obtaining an odd number"}$. $A^\star = \{2,4,6\}$.

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Example: Venn diagrams for event composition
#
# ![Venn diagrams for events composition](_images/event_composition_venn_diagram.png)
#
# - (i): $B \cap C$
# - (ii): $B \cup C$
# - (iii): $A^\star \cap B$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Event algebra
#
# * Idempotency: $A \cup A = A$ and $A \cap A = A$
# * Commutativity: $A \cup B = B \cup A$ and $A \cap B = B \cap A$
# * Associativity: $(A \cup B) \cup C = A \cup (B \cup C) = A \cup B \cup C$ and $(A \cap B) \cap C = A \cap (B \cap C) = A \cap B \cap C$
# * Distributivity: $(A \cup B) \cap C = (A \cap C) \cup (B \cap C)$ and $(A \cap B) \cup C = (A \cup B) \cap (B \cup C)$
# * Identities : $A \cup \emptyset = A$, $A \cup \Omega = \Omega$, $A \cap \emptyset = \emptyset$ and $A \cap \Omega = A$
# * Complementarity: $A \cup A^* = \Omega$, $A \cap A^* = \emptyset$, $(A^*)^* = A$, $\emptyset^* = \Omega$ and $\Omega^* = \emptyset$
# * De Morgan laws:
#   * $(A \cup B)^* = A^* \cap B^*$ and $(A \cap B)^* = A^* \cup B^*$
#   * $(\bigcup_{i=1}^{n} A_i)^\star = \bigcap_{i=1}^{n} A_i^\star$ and $(\bigcap_{i=1}^{n} A_i)^\star = \bigcup_{i=1}^{n} A_i^\star$

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Example: Venn diagrams for events algebra
#
# ![Venn diagrams for events albegra](_images/event_algebra_venn_diagram.png)
#
# - (i): $(B \cap C)^\star = B^\star \cup C^\star$
# - (ii): $((A \cap B^\star) \cup C^\star)^\star = (A^\star \cap C) \cup (B \cap C)$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Event partition
#
# An event set $\{ A_1, \dots, A_n \}$ is a **partition** of the sample space if and only if:
#
# * No event is impossible: $\forall i, A_i \ne \emptyset$
# * Events are incompatible two by two: $\forall i \ne j, A_i \cap A_j = \emptyset$
# * Union of all events is equal to the sample space: $\bigcup_{i=1}^{n} A_i = \Omega$
#
# ![Event partition](_images/event_partitions.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Combination of random experiments
#
# For two joint experiments characterized by sample spaces $\Omega_1$ and $\Omega_2$, we can define a new sample space $\Omega$ as their cartesian product: $\Omega = \Omega_1 \times \Omega_2$
#
# Example for rolling two 6-sided dices: $\Omega_1 = \Omega_2 = \{1,2,3,4,5,6\}$ and $\Omega = \{(1,1), (1,2), \dots, (6,5), (6,6) \}$ (36 couples).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Probability basics

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Probability of an event
#
# #### Classical definition
#
# > "The probability of an event is the ratio of the number of cases favorable to it, to the number of all cases possible when nothing leads us to expect that any one of these cases should occur more than any other, which renders them, for us, equally possible." (Laplace, 1819)
#
# $$P(A) = \frac{\# A}{\# \Omega}$$
#
# Assumes a finite number of cases and equiprobability between them.

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Example: throwing dices
#
# - Throwing a 6-sided dice. $\Omega = \{1,2,3,4,5,6\}$. $A \equiv$ "obtaining 5".
#
# $$P(A) = \frac{\# A}{\# \Omega} = \frac{1}{6}$$
#
# - Throwing two 6-sided dices and summing their values. $\Omega = \{2,3,\dots,12\}$. $A \equiv$ "obtaining 5".
#
# $$P(A) \neq \frac{\# A}{\# \Omega} = \frac{1}{11} \; \text{(no equiprobability)}$$
#
# $$P(A) = \frac{\# A'}{\# \Omega'} = \frac{4}{36} = \frac{1}{6}$$
#
# With $A' = \{(1,4), (2,3), (3,2), (4,1)\}$ and $\Omega'=\{(1,1), (1,2), \dots, (6,6)\}$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Frequentist definition
#
# For $n$ repetitions under the same conditions of a random experiment, $P(A)$ is the theorical frequency of event $A$.
#
# $$P(A) = \lim_{n\to \infty} \frac{\# A}{n}$$
#
# Assumes the possibility of indefinitely repeating an experiment without changing its conditions.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Axiomatic definition
#
# A **probability** is a number that satisfies the following axioms (Kolmogorov, 1933):
#
# * For any event $A$, $P(A) \ge 0$
# * $P(\Omega) = 1$
# * If events $A$ and $B$ are incompatible ($A \cap B = \emptyset $), then $P(A \cup B) = P(A)+P(B)$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Properties of probabilities
#
# The previous definition has the following consequences:
#
# * $P(\emptyset) = 0$
# * $P(A) \le 1$
# * $P(A^*) = 1 - P(A)$
# * If $\forall i \ne j, A_i \cap A_j = \emptyset$, then $P(\bigcup_{i=1}^{n} A_i) = \sum_{i=1}^{n} P(A_i)$
# * $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
# * In the general case, $P(\bigcup_{i=1}^{n} A_i) \le \sum_{i=1}^{n} P(A_i)$

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Conditional probability
#
# The probability $P(A|B)$ of event $A$ knowing that event $B$ has occurred is given by:
#
# $$P(A|B) = \frac{P(A \cap B)}{P(B)} \qquad \text{with} \space P(B) \ne 0$$
#
# ![Conditional probability](_images/conditional_proba.png)
#
# There is no fundamental difference between conditional and non-conditional probabilities: $P(A) = P(A|\Omega)$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Example: throwing a 6-sided dice
#
# $A \equiv$ "result is $\leq$ 3", $B \equiv$ "result is even", $C \equiv$ "result = 6".
#
# ![Probabilities for A,B and C](_images/probas_d6.png)
#
# - $P(A) = P(B) = \frac{1}{2}$ even though $A \neq B$.
# - $P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{1}{3}$.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Properties of conditional probabilities
#
# The previous definition has the following consequences:
#
# * $P(\Omega|B) = 1$
# * $P(\emptyset|B) = 0$
# * $P(A^*|B) = 1 - P(A|B)$
# * If $\forall i \ne j, A_i \cap A_j = \emptyset$, then $P(\bigcup_{i=1}^{n} A_i|B) = \sum_{i=1}^{n} P(A_i|B)$
# * $P((A \cup C)|B) = P(A|B)+P(C|B)-P((A \cap C)|B)$
# * If $B \subset A$, $P(A|B) = 1$
# * If $A \subset B$, $P(A|B) = \frac{P(A)}{P(B)} \ge P(A)$

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Fundamental probability laws
#
# #### General product rule (a.k.a. chain rule)
#
# * $P(A \cap B) = P(A|B)P(B)$
# * More generally, for any events $A_1, A_2,\dots,A_n$:
#
# $$P(A_1 \cap \cdots \cap A_n ) = P(A_1)P(A_2|A_1)P(A_3|(A_1 \cap A_2)) \dots P(A_n|(A_1 \cap \cdots \cap A_{n-1}))$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Law of total probability
#
# If the set of events $\{B_1, B_2, \dots, B_n\}$ is a partition of $\Omega$, then:
#
# $$P(A) = \sum_{i=1}^{n} P(A|B_i)P(B_i)$$
#
# ![Law of total probability](_images/law_total_probas.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Bayes' theorem
#
# For any two events $A$ and $B$:
#
# $$P(B|A) = \frac{P(A|B)P(B)}{P(A)}$$
#
# If the set of events $\{B_1, B_2, \dots, B_n\}$ is a partition of $\Omega$, then:
#
# $$P(B_i|A) = \frac{P(A|B_i)P(B_i)}{\sum_{j=1}^{n} P(A|B_j)P(B_j)}$$
#
# $P(B_i)$ is the *prior* probability (known before the random experiment). $P(B_i|A)$ is the *posterior* probability.
#
# The $B_i$ events can be seen as the possible causes responsible for the occurrence of $A$.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Independance between events
#
# For any two events $A$ and $B$:
#
# $$A \perp B \Longleftrightarrow P(A \cap B) = P(A)P(B) \Longleftrightarrow P(A|B)=P(A) \; \text{with} \; P(B) \neq 0$$
#
# For any events $A_1,\dots,A_n$:
#
# $$A_1 \perp \dots \perp A_n \Longleftrightarrow P (\bigcap_{j \in J} A_j) = \prod_{j \in J} P(A_j) \; \text{for every} \; J \subseteq \{1,\dots,n\}$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Conditional independance
#
# For any three events $A$, $B$ and $C$:
#
# $$(A \perp B)|C \Longleftrightarrow P((A \cap B)|C) = P(A|C)P(B|C)$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Independance between trials
#
# Two experiments characterized by sample spaces $\Omega_1$ and $\Omega_2$ are independant if and only if:
#
# $$P(A \times B) = P(A)P(B) \qquad \forall A \in \Omega_1, B \in \Omega_2$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Example : throwing a 6-sided dice twice
#
# $A_i \equiv$ "obtaining number $i$ at first throw", $B_j \equiv$ "obtaining number $j$ at second throw".
#
# $$P(A_i \times B_j) = P(A_i)P(B_j) = \frac{1}{6} \times \frac{1}{6} = \frac{1}{36}$$

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Random variables

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Definition
#
# A random varialbe $X$ is an application from $\Omega$ to $\mathbb{R}$ that associates a value $x=X(\omega) \in \mathbb{R}_X$ to each elementary event $\omega \in \Omega$.
#
# $\mathbb{R}_X \subseteq \mathbb{R}$ is called the *variation domain* of $X$.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Example: throwing a 6-sided dice
#
# The financial gain can be defined as a random variable $X$.
#
# $\mathbb{R}_X = \{-5,0,5\}$
#
# ![Random variable example](_images/random_variable_example.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Discrete random variables
#
# #### Probability function
#
# It associates to each value $x$ the probability $p(x)$ that the random variable $X$ takes this value.
#
# $$p(x) =
#     \begin{cases}
#       P(X=x) \qquad \text{if} \; x \in \mathbb{R}_X \\
#       0 \qquad \qquad \;\;\;\; \text{if} \; x \notin \mathbb{R}_X
#     \end{cases}$$
#     
# $$\forall x \in \mathbb{R}, p(x) \ge 0$$
#
# $$\sum_{i=1}^n p(x_i) = \sum_{i=1}^n P(X=x_i) = P(\bigcup_{i=1}^n X = x_i) = 1$$

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Example 1: throwing a 6-sided dice twice (followup)
#
# |$x$|$-5$|$0$|$5$|
# |-|-|-|-|
# |$p(x)$|$\frac{1}{6}$|$\frac{2}{3}$|$\frac{1}{6}$|

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Example 2: number of major earthquakes in a century
#
# ![Probability function example](_images/proba_function_example.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Probability of an event related to a random variable
#
# For an event $B \subseteq \mathbb{R}_X$, its probability is given by:
#
# $$P(B) = \sum_{x \in B} p(x)$$

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Example: throwing a 6-sided dice twice (followup)
#
# $B \equiv$ "not losing money".
#
# $$P(B) = P(X \ge 0) = \frac{5}{6}$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Distribution function
#
# $$F(x) = P(X \le x) = \sum_{x_i \le x} p(x_i) \qquad \forall x \in \mathbb{R}, x_i \in \mathbb{R}_X$$
#
# $$\forall x_i \lt x_j, F(x_i) \le F(x_j)$$
#
# $F(x)$ is a monotonically increasing "staircase" function.
#
# ![Distribution function example](_images/distri_function_example.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Bernoulli's law
#
# We only consider the occurence (or lack thereof) of an event $A$, given by its probability $p = P(A)$.
#
# $$\Omega = \{A, A^*\}$$
#
# The random variable $X$ associating $1$ to the occurence of $A$ and $0$ otherwise is called Bernoulli's law.
#
# $$X \sim Be(p) \Longleftrightarrow p(x) =
#     \begin{cases}
#       p \qquad \text{if} \; x=1 \qquad \text{(}A \text{ occurred)}\\
#       1-p \;\; \text{if} \; x=0 \qquad \text{(}A^* \text{ occurred)}\\
#       0 \qquad \text{otherwise}
#     \end{cases}$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Binomial law
#
# A Bernoulli trial is repeated several times.
#
# If $Y_i \sim \dots \sim Y_n \sim Be(p)$ with $Y_1 \perp \dots \perp Y_n$, then:
#
# $$X = \sum_{i=1}^n Y_i \sim Bi(n,p)$$
#
# $$X \sim Bi(n,p) \Longleftrightarrow p(x) =
#     \begin{cases}
#       C_n^x p^x(1-p)^{n-x} \qquad \text{if } x \in \{0, \dots, n\}\\
#       0 \qquad \qquad \qquad \;\;\;\; \text{otherwise}
#     \end{cases}$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Hypergeometric law
#
# The binomial law describes the probability of $x$ successes (random draws for which the object drawn has a specified feature) in $n$ draws with replacement.
#
# In contrast, the hypergeometric law describes the probability of $x$ successes in $n$ draws *without replacement*, from a finite population of size $N$ that contains exactly $k$ objects with that feature. As such, there is no independance between the successive results of the $n$ Bernoulli trials.
#
# $$X \sim Hy(N,n,k) \Longleftrightarrow p(x) =
#     \begin{cases}
#       \frac{C_k^x C_{N-k}^{n-x}}{C_N^n} \qquad \text{with max}(0, n+k-N) \le x \le \text{min}(n,k)\\
#       0 \qquad \qquad \; \text{otherwise}
#     \end{cases}$$
#     
# When $N \gg n$, it can be approximated by $Bi(n, k / N)$.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Geometric law
#
# This law describes the probability of obtaining a first success when repeating independant Bernoulli trials.
#
# $$X \sim Ge(p) \Longleftrightarrow p(x) =
#     \begin{cases}
#       p(1-p)^{x-1} \qquad \text{if } x \in \mathbb{N}^+\\
#       0 \qquad \qquad \;\;\;\;\;\; \text{otherwise}
#     \end{cases}$$

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Appendices

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Combinations
#
# $$C_n^k = \frac{n!}{k!(n-k)!}$$

# %% slideshow={"slide_type": "slide"}

# https://www.tensorflow.org/guide/tensor
# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
# https://en.wikipedia.org/wiki/Event_(probability_theory)
# https://devmath.fr/tools/latex-symbols-list/#
#

# %%
