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
# # CIFAR10
#
# ## Objective
#
# Training models to associate images representing common objects with their class (multiclass classification).
#
# ## Context
#
# The [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are completely mutually exclusive. There are 50,000 training images and 10,000 test images.
#
# ![CIFAR10 images](_images/cifar10.png)
#
# ## Instructions and advice
#
# - Follow the main steps of a supervised ML project: data loading and exploring, data preparation, model training and evaluation.
# - Use the [PyTorch](https://pytorch.org) library for data loading and model training. If you are new to it, consider following its [official tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html).
# - Don't forget to setup your environment by importing the necessary Python packages. Several helper functions (see below) have already been defined for you.
# - Regarding data loading and preparation, bitmap images should be normalized. You can find an example using the CIFAR10 dataset [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#load-and-normalize-cifar10).
# - You may train and evaluate a standard MultiLayer Perceptron, using [this chapter](../ann.ipynb) as a blueprint. Warning: you will have to adapt the inputs of your network to the color images of the CIFAR10 dataset.
# - **Bonus**: train a convolutional neural network using [this chapter](../cnn.ipynb) as a blueprint. After training, compare its performance with the MLP results.

# %% [markdown]
# ## Environment setup

# %% slideshow={"slide_type": "skip"}
# pylint: disable=wrong-import-position

import os

# Installing the ainotes package is only necessary in standalone runtime environments like Colab
if os.getenv("COLAB_RELEASE_TAG"):
    print("Standalone runtime environment detected, installing ainotes package")
    # %pip install ainotes

# pylint: enable=wrong-import-position

# %%
import platform

import numpy as np
import seaborn as sns

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ainotes.utils.plot import plot_loss_acc, plot_cifar10_images
from ainotes.utils.train import get_device, count_parameters, fit

# %%
# Setup plots

# Include matplotlib graphs into the notebook, next to the code
# https://stackoverflow.com/a/43028034/2380880
# %matplotlib inline

# Improve plot quality
# %config InlineBackend.figure_format = "retina"

# Setup seaborn default theme
# http://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
sns.set_theme()


# %%
# Print environment info
print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")


# PyTorch device configuration
device, message = get_device()
print(message)

# %% [markdown]
# ## Data loading and exploring

# %%
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # Dataset contains PIL images of range [0, 1]
        # We transform them to Tensors of normalized range [-1, 1]
        # https://stackoverflow.com/a/65679179
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

cifar_train_data = datasets.CIFAR10(
    root="data", train=True, download=True, transform=transform
)

cifar_test_data = datasets.CIFAR10(
    root="data", train=False, download=True, transform=transform
)

cifar_labels = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# %%
cifar_img, cifar_label = cifar_train_data[0]

print(f"First image: {cifar_img.shape}. Label: {cifar_label}")

# Unnormalize pixel values
pixel_values = (cifar_img[:, 0, 0] / 2 + 0.5) * 255
print(f"RGB values of first pixel: {pixel_values.int()}")

# %%
# Plot some of the training images
plot_cifar10_images(data=cifar_train_data, labels=cifar_labels, device=device)


# %% [markdown]
# ## Hyperparameters

# %%
# Hyperparameters
learning_rate = 1e-3
n_epochs = 10
batch_size = 64

# %% [markdown]
# ## Data preparation

# %%
cifar_train_dataloader = DataLoader(cifar_train_data, batch_size=batch_size)
cifar_test_dataloader = DataLoader(cifar_test_data, batch_size=batch_size)


# %% [markdown]
# ## Training a MLP
#
# ### Model definition


# %%
class NeuralNetwork(nn.Module):
    """Neural network for CIFAR images classification"""

    def __init__(self):
        super().__init__()

        # Flatten the input image of shape (3, 32, 32) into a vector of shape (3*32*32,)
        self.flatten = nn.Flatten()

        # Define a sequential stack of linear layers and activation functions
        self.layer_stack = nn.Sequential(
            # First hidden layer with 3072 inputs
            nn.Linear(in_features=3 * 32 * 32, out_features=64),
            nn.ReLU(),
            # Second hidden layer
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            # Output layer
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        """Define the forward pass of the model"""

        # Apply flattening to input
        x = self.flatten(x)

        # Compute output of layer stack
        logits = self.layer_stack(x)

        # Logits are a vector of raw (non-normalized) predictions
        # This vector contains 10 values, one for each possible class
        return logits


# %%
cifar_mlp = NeuralNetwork().to(device)
print(cifar_mlp)

# Try to guess the total number of parameters for this model before running this code!
print(f"Number of trainable parameters: {count_parameters(cifar_mlp)}")


# %% [markdown]
# ### Model training


# %%
cifar_mlp_history = fit(
    dataloader=cifar_train_dataloader,
    model=cifar_mlp,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(cifar_mlp.parameters(), lr=learning_rate),
    epochs=n_epochs,
    device=device,
)

# %% [markdown]
# ### Training results

# %%
# Plot training history
plot_loss_acc(cifar_mlp_history)

# %%
# Show model predictions on some test images
plot_cifar10_images(
    data=cifar_train_data, labels=cifar_labels, device=device, model=cifar_mlp
)


# %% [markdown]
# ### Results interpretation
#
# Although training smoothly, the MLP model incorrectly classifies more than 1/3 of the training data. Its architecture is too limited for this non-trivial task.

# %% [markdown]
# ## Training a convnet
#
# ### Model definition


# %%
class Convnet(nn.Module):
    """Convnet for CIFAR image classification"""

    def __init__(self):
        super().__init__()

        # Define a sequential stack
        self.layer_stack = nn.Sequential(
            # Feature extraction part, with convolutional and pooling layers
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Classification part, with fully connected layers
            nn.Flatten(),
            nn.Linear(in_features=2304, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):
        """Define the forward pass of the model"""

        # Compute output of layer stack
        logits = self.layer_stack(x)

        # Logits are a vector of raw (non-normalized) predictions
        # This vector contains 10 values, one for each possible class
        return logits


# %%
cifar_convnet = Convnet().to(device)
print(cifar_convnet)

# Try to guess the total number of parameters for this model before running this code!
print(f"Number of trainable parameters: {count_parameters(cifar_convnet)}")

# %% [markdown]
# ### Model training

# %%
cifar_convnet_history = fit(
    dataloader=cifar_train_dataloader,
    model=cifar_convnet,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(cifar_convnet.parameters(), lr=learning_rate),
    epochs=n_epochs,
    device=device,
)

# %% [markdown]
# ### Training results

# %%
# Plot training history
plot_loss_acc(cifar_convnet_history)

# %%
# Show model predictions on some test images
plot_cifar10_images(
    data=cifar_train_data, labels=cifar_labels, device=device, model=cifar_convnet
)

# %% [markdown]
# ### Results interpretation
#
# The convnet is much better than the MLP at classifying CIFAR10 images, which is not surprising given that its architecture was invented for this kind of task.
#
# Results could probably be even better with a longer training time.

# %%
