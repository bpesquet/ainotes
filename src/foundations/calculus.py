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
# # Calculus

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
# To be continued...
