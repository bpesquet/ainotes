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
# # Data Manipulation

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %%
import platform
import torch

print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")

# %% slideshow={"slide_type": "slide"}
# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA GPU {torch.cuda.get_device_name(0)} found :)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Metal GPU found :)")
else:
    device = torch.device("cpu")
    print("No available GPU :/")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Working with tensors

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Definition
#
# In the context of AI, a **tensor** is a set of primitive values sharing the same type (most often numerical), shaped into an array of any number of dimensions. It is a fancy name for a multidimensional array.
#
# Tensors are heavily used by AI algorithms to represent and manipulate information. They are, in particular, the core data structures of Machine Learning.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Tensor properties
#
# - A tensor's dimension is also called an **axis**.
# - A tensor's **rank** is its number of axes.
# - The tensor's **shape** describes the number of values along each axis.
#
# In mathematical terms, a rank 0 tensor is a **scalar**, a rank 1 tensor is a **vector** and a rank 2 tensor is a **matrix**.
#
# > Warning: *rank* and *dimension* are polysemic terms, which can be confusing.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Tensors in Python
#
# Python offers limited native support for manipulating tensors. Lists can be used to store information, but their mathematical capacities are insufficient for any serious work.

# %%
# A vector (rank 1 tensor)
a = [1, 2, 3]
print(a)

# A matrix (rank 2 tensor)
b = [a, [4, 5, 6]]
print(b)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Dedicated libraries
#
# Over the years, several tools have been developed to overcome Python's native limitations. One of the most widely used is [NumPy](https://numpy.org/), which supports tensors in the form of `ndarray` objects. It offers a comprehensive set of operations on them, including creating, sorting, selecting, linear algebra and statistical operations.
#
# For all its qualities, NumPy has a limitation which can be critical in some contexts: it only runs on the machine's [CPU](https://en.wikipedia.org/wiki/Processor_(computing)). Among other advantages, newer tools offer support for dedicated high-performance processors like [GPUs](https://en.wikipedia.org/wiki/Graphics_processing_unit) or [TPUs](https://en.wikipedia.org/wiki/Tensor_Processing_Unit), while providing a NumPy-like API to make onboarding easier. The most prominent ones are currently [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org) and [JAX](https://jax.readthedocs.io).
#
# This content uses PyTorch, which strikes a good balance between power, flexibility and user-friendliness.

# %%
