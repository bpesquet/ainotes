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
# ## Probability

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Fundamentals
#
# #### Random experiment
#
# Experiment implying some randomness. Knowing the experimental conditions is not enough to predict its outcome. Opposite of *deterministic experiment*.
#
# Example: throwing a six-sided dice.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Sample space
#
# Set of all possible outcomes for the random experiment, denoted by $\Omega$ or $S$. Can either be finite or infinitely large.
#
# Example: $\Omega = \{1,2,3,4,5,6\}$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Event
#
# Subset of the sample space, denoted by $E \subseteq \Omega$.
#
# An event is a set of elementary events (observed outcomes). An elementary event is denoted $\omega \in \Omega$. If $\omega \in E$, then event $E$ has occurred.
#
# Example: $E \equiv \text{"obtaining an odd number"}$. $E= \{1,3,5\}$

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
# * Negation $E^\star = \Omega \backslash E$. Occurs if and only if $E$ does not occur.
# * Conjunction $A \cap B$. Occurs if and only if $A$ and $B$ occur simultaneously.
# * Union $A \cup B$. Occurs if either $A$ or $B$ occurs.
# * Generalization to $n$ events:
#   * $\bigcap_{i=1}^{n} A_i =  A_1 \cap A_2 \cap \cdots \cap A_n$
#   * $\bigcup_{i=1}^{n} A_i =  A_1 \cup A_2 \cup \cdots \cup A_n$
# * $(\bigcap_{i=1}^{n} A_i) \subseteq (\bigcup_{i=1}^{n} A_i)$ is always verified.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Event algebra
#
# * Idempotency: $A \cup A = A$ and $A \cap A = A$
# * Commutativity: $A \cup B = B \cup A$ and $A \cap B = B \cap A$
# * Associativity: $(A \cup B) \cup C = A \cup (B \cup C) = A \cup B \cup C$ and $(A \cap B) \cap C = A \cap (B \cap C) = A \cap B \cap C$
# * Distributivity: $(A \cup B) \cap C = (A \cap C) \cup (B \cap C)$ and $(A \cap B) \cup C = (A \cup B) \cap (B \cup C)$
# * Identities : $A \cup \emptyset = A$, $A \cup \Omega = \Omega$, $A \cap \emptyset = \emptyset$ and $A \cap \Omega = A$
# * Complementarity: $A \cup A^* = \Omega$, $A \cap A^* = \emptyset$, $(A^*)^* = A$, $\emptyset^* = \Omega$ and $\Omega^* = \emptyset$
# * De Morgan laws: $(A \cup B)^* = A^* \cap B^*$ and $(A \cap B)^* = A^* \cup B^*$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Event partition
#
# An event set $\{ A_1, \dots, A_n \}$ is a partition of the sample space if and only if:
#
# * No event is impossible: $\forall i, A_i \ne \emptyset$
# * Events are incompatible two by two: $\forall i \ne j, A_i \cap A_j = \emptyset$
# * Union of all events is equal to the sample space: $\bigcup_{i=1}^{n} A_i = \Omega$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Combination of random experiments
#
# For two joint experiments characterized by sample spaces $\Omega_1$ and $\Omega_2$, we can define a new sample space $\Omega$ as their cartesian product: $\Omega = \Omega_1 \times \Omega_2$
#
# Example for rolling two dices: $\Omega_1 = \Omega_2 = \{1,2,3,4,5,6\}$ and $\Omega = \{(1,1), (1,2), \dots, (6,5), (6,6) \}$ (36 couples).

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Probability basics
#
# #### Classical definition
#
# The probability of an event is the ratio of the number of cases favorable to it, to the number of all cases possible when nothing leads us to expect that any one of these cases should occur more than any other, which renders them, for us, equally possible. (Laplace, 1819)
#
# $$P(A) = \frac{\# A}{\# \Omega}$$
#
# Assumes equiprobability and a finite number of cases.

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
# A probability is a number that satisfies the following axioms (Kolmogorov, 1933):
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
# * $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
# * If $\forall i \ne j, A_i \cap A_j = \emptyset$, then $P(\bigcup_{i=1}^{n} A_i) = \sum_{i=1}^{n} P(A_i)$
# * In the general case, $P(\bigcup_{i=1}^{n} A_i) \le \sum_{i=1}^{n} P(A_i)$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Conditional probability
#
# The probability $P(A|B)$ of event $A$ knowing that event $B$ has occurred is given by:
#
# $$P(A|B) = \frac{P(A \cap B)}{P(B)} \qquad \text{with} \space P(B) \ne 0$$
#
# There is no fundamental difference between conditional and non-conditional probabilities: $P(A) = P(A|\Omega)$

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
# #### Chain rule
#
# * $P(A \cap B) = P(A|B)P(B)$
# * More generally, for any events $A_1, A_2,\dots,A_n$,
#
# $$P(A_1 \cap \cdots \cap A_n ) = P(A_1)P(A_2|A_1)P(A_3|(A_1 \cap A_2)) \dots P(A_n|(A_1 \cap \cdots \cap A_{n-1}))$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Law of total probability
#
# If the set of events $\{B_1, B_2, \dots, B_n\}$ is a partition of $\Omega$, then:
#
# $$P(A) = \sum_{i=1}^{n} P(A|B_i)P(B_i)$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Bayes' theorem
#
# If the set of events $\{B_1, B_2, \dots, B_n\}$ is a partition of $\Omega$, then:
#
# $$P(B_i|A) = \frac{P(A|B_i)P(B_i)}{\sum_{i=1}^{n} P(A|B_i)P(B_i)}$$
#
# $P(B_i)$ is the prior probability (known before the random experiment). $P(B_i|A)$ is the posterior probability.
#
# The $B_i$ events can be seen as the possible causes responsible for the occurrence of $A$.

# %% [markdown] slideshow={"slide_type": "slide"}
# To be continued...

# %% slideshow={"slide_type": "slide"}

# https://www.tensorflow.org/guide/tensor
# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
# https://en.wikipedia.org/wiki/Event_(probability_theory)
# https://devmath.fr/tools/latex-symbols-list/#
#
