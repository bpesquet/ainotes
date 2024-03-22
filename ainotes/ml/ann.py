# %% [markdown] slideshow={"slide_type": "slide"}
# # Artificial neural networks

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Learning objectives
#
# - Know the possibilities, architecture and key components of an artificial neural network.
# - Understand how neural networks are trained.
# - Learn how to build neural networks with [PyTorch](https://pytorch.org/).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "skip"}
# pylint: disable=wrong-import-position

import os

# Installing the ainotes package is only necessary in standalone runtime environments like Colab
if os.getenv("COLAB_RELEASE_TAG"):
    print("Standalone runtime environment detected, installing ainotes package")
    # %pip install ainotes

# pylint: enable=wrong-import-position

# %% slideshow={"slide_type": "slide"}
import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

import sklearn
from sklearn.datasets import make_circles

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from ainotes.utils.plot import plot_loss_acc, plot_fashion_images
from ainotes.utils.train import get_device, count_parameters, fit

# %% slideshow={"slide_type": "slide"}
# Setup plots

# Include matplotlib graphs into the notebook, next to the code
# https://stackoverflow.com/a/43028034/2380880
# %matplotlib inline

# Improve plot quality
# %config InlineBackend.figure_format = "retina"

# Setup seaborn default theme
# http://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
sns.set_theme()


# %% slideshow={"slide_type": "skip"}
# Utility functions


def plot_activation_function(f, f_prime, name, axis=(-6, 6, -1.1, 1.1)):
    """Plot an activation function and its derivative"""

    x_min, x_max = axis[0], axis[1]

    z = np.linspace(x_min, x_max, 200)
    plt.plot(z, f(z), "b-", linewidth=2, label=name)
    plt.plot(z, f_prime(z), "g--", linewidth=2, label=f"{name}'")
    plt.xlabel("x")
    plt.ylabel(f"{name}(x)")
    plt.axis(axis)
    plt.legend(loc="upper left")
    plt.show()


def plot_dataset(x, y):
    """Plot a 2-dimensional dataset with associated classes"""

    plt.figure()
    plt.plot(x[y == 0, 0], x[y == 0, 1], "or", label=0)
    plt.plot(x[y == 1, 0], x[y == 1, 1], "ob", label=1)
    plt.legend()
    plt.show()


def plot_decision_boundary(model, x, y):
    """Plot the frontier between classes for a 2-dimensional dataset"""

    plt.figure()

    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Compute model output for the whole grid
    z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device))
    z = z.reshape(xx.shape)
    # Convert PyTorch tensor to NumPy
    zz = z.cpu().detach().numpy()
    # Plot the contour and training examples
    plt.contourf(xx, yy, zz, cmap=plt.colormaps.get_cmap("Spectral"))
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright)
    plt.show()


# %% slideshow={"slide_type": "slide"}
# Print environment info
print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"PyTorch version: {torch.__version__}")


# PyTorch device configuration
# Performance issues exist with MPS backend for MLP-like models
device, message = get_device(use_mps=False)
print(message)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Fundamentals

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Online playground
#
# [![TensorFlow playground](_images/tf_playground.png)](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.1&regularizationRate=0&noise=0&networkShape=2&seed=0.59857&showTestData=false&discretize=false&percTrainData=30&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&showTestData_hide=false&problem_hide=true&regularization_hide=true&regularizationRate_hide=true&percTrainData_hide=false)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### History
#
# #### A biological inspiration
#
# ![Neuron](_images/neuron.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### McCulloch & Pitts' formal neuron (1943)
#
# ![Formal neuron model](_images/neuron_model.jpeg)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Hebb's rule (1949)
#
# Attempt to explain synaptic plasticity, the adaptation of brain neurons during the learning process.
#
# > "The general idea is an old one, that any two cells or systems of cells that are repeatedly active at the same time will tend to become 'associated' so that activity in one facilitates activity in the other."

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Franck Rosenblatt's perceptron (1958)
#
# ![The Perceptron](_images/Perceptron.jpg)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### The perceptron learning algorithm
#
# 1. Init randomly the connection weights $\pmb{\omega}$.
# 1. For each training sample $\pmb{x}^{(i)}$:
#     1. Compute the perceptron output $y'^{(i)}$
#     1. Adjust weights : $\pmb{\omega_{t+1}} = \pmb{\omega_t} + \eta (y^{(i)} - y'^{(i)}) \pmb{x}^{(i)}$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Minsky's critic (1969)
#
# One perceptron cannot learn non-linearly separable functions.
#
# ![XOR problem](_images/xor.png)
#
# At the time, no learning algorithm existed for training the hidden layers of a MLP.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Decisive breakthroughs (1970s-1990s)
#
# - 1974: backpropagation theory (P. Werbos).
# - 1986: learning through backpropagation (Rumelhart, Hinton, Williams).
# - 1989: first researchs on deep neural nets (LeCun, Bengio).
# - 1991: Universal approximation theorem. Given appropriate complexity and appropriate learning, a network can theorically approximate any continuous function.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Key components
#
# #### Anatomy of a fully connected network
#
# ![A neural network](_images/nn_weights.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Neuron output
#
# ![Neuron output](_images/neuron_output.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Activation functions
#
# They are applied to the weighted sum of neuron inputs to produce its output.
#
# They must be:
#
# - **non-linear**, so that the network has access to a richer representation space and not only linear transformations;
# - **differentiable**, so that gradients can be computed during learning.

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### Sigmoid
#
# This function "squashes" its input between between 0 and 1, outputting something that can be interpreted as the probability of the positive class. It is often used in the final layer of the network for binary classification tasks.
#
# $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
#
# $$\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \sigma(x)\big(1 - \sigma(x)\big)$$


# %% slideshow={"slide_type": "slide"}
def sigmoid(x):
    """Sigmoid function"""

    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """Derivative of the sigmoid function"""

    return sigmoid(x) * (1 - sigmoid(x))


# %% slideshow={"slide_type": "-"}
plot_activation_function(sigmoid, sigmoid_prime, "Sigmoid", axis=(-6, 6, -0.1, 1.1))

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### tanh
#
# The hyperbolic tangent function has a similar shape as sigmoid, but outputs values in the $[-1,1]$ interval.
#
# $$tanh(x) = \frac{sinh(x)}{cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{2}{1+e^{-2x}} -1 = 2\sigma(2x) - 1$$
#
# $$tanh'(x) = \frac{4}{(e^x + e^{-x})^2} = \frac{1}{cosh^2(x)}$$


# %% slideshow={"slide_type": "slide"}
def tanh(x):
    """Hyperbolic tangent function"""

    return 2 * sigmoid(2 * x) - 1


def tanh_prime(x):
    """Derivative of hyperbolic tangent function"""

    return 4 / np.square(np.exp(x) + np.exp(-x))


# %% slideshow={"slide_type": "-"}
plot_activation_function(tanh, tanh_prime, "Tanh", axis=(-6, 6, -1.1, 1.1))

# %% [markdown] slideshow={"slide_type": "slide"}
# ##### ReLU
#
# The Rectified Linear Unit function has replaced sigmoid and tanh as the default activation function in most contexts.
#
# $$ReLU(x) = max(0,x)$$
#
# $$ReLU'(x) =
#     \begin{cases}
#       0 \; \forall x \in\; ]-\infty, 0] \\
#       1 \; \forall x \in\; ]0, +\infty[
#     \end{cases}$$


# %% slideshow={"slide_type": "slide"}
def relu(x):
    """Rectified Linear Unit function"""

    return np.maximum(0, x)


def relu_prime(x):
    """Derivative of the Rectified Linear Unit function"""

    # https://stackoverflow.com/a/45022037
    return (x > 0).astype(x.dtype)


# %% slideshow={"slide_type": "-"}
plot_activation_function(relu, relu_prime, "ReLU", axis=(-6, 6, -1.1, 6.1))

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Training process
#
# #### Learning algorithm
#
# [![Extract from the book Deep Learning with Python](_images/nn_learning.jpg)](https://www.manning.com/books/deep-learning-with-python)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Weights initialization
#
# To facilitate training, initial weights must be:
#
# - non-zero
# - random
# - have small values
#
# [Several techniques exist](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79). A commonly used one is [Xavier initialization](https://proceedings.mlr.press/v9/glorot10a.html).

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Weights update
#
# Objective: minimize the loss function. Method: [gradient descent](principles.ipynb).
#
# $$\pmb{\omega_{t+1}} = \pmb{\omega_t} - \eta\nabla_{\pmb{\omega}}\mathcal{L}(\pmb{\omega_t})$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Backpropagation
#
# Objective: compute $\nabla_{\pmb{\omega}}\mathcal{L}(\pmb{\omega_t})$, the loss function gradient w.r.t. all the network weights.
#
# Method: apply the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) to compute partial derivatives backwards, starting from the current output.
#
# $$y = f(g(x)) \;\;\;\; \frac{\partial y}{\partial x} = \frac{\partial f}{\partial g} \frac{\partial g}{\partial x}\;\;\;\; \frac{\partial y}{\partial x} = \sum_{i=1}^n \frac{\partial f}{\partial g^{(i)}} \frac{\partial g^{(i)}}{\partial x}$$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Visual demo of backpropagation
#
# [![Backprop explained visually](_images/visual_backprop_demo.png)](https://developers-dot-devsite-v2-prod.appspot.com/machine-learning/crash-course/backprop-scroll)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## BInary classification example

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Data generation and visualization
#
# A scikit-learn [function](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html) is used to easily generate two-dimensional data with two classes.

# %% slideshow={"slide_type": "-"}
# Generate 2D data (a large circle containing a smaller circle)
planar_data, planar_targets = make_circles(n_samples=500, noise=0.1, factor=0.3)

print(f"Data: {planar_data.shape}. targets: {planar_targets.shape}")
print(planar_data[:10])
print(planar_targets[:10])


# %% slideshow={"slide_type": "slide"}
plot_dataset(planar_data, planar_targets)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Hyperparameters
#
# Hyperparameters ($\neq$ model parameters) are adjustable configuration values that let you control the model training process.

# %% slideshow={"slide_type": "-"}
# Rate of parameter change during gradient descent
learning_rate = 0.1

# An epoch is finished when all data samples have been presented to the model during training
n_epochs = 50

# Number of samples used for one gradient descent step during training
batch_size = 5

# Number of neurons on the hidden layer of the MLP
hidden_layer_size = 2

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Data preparation
#
# Generated data (NumPy tensors) needs to be converted to PyTorch tensors before training a PyTorch-based model. These new tensors are stored in the memory of the available device (GPU ou CPU).

# %% slideshow={"slide_type": "-"}
# Create PyTorch tensors from NumPy tensors

x_train = torch.from_numpy(planar_data).float().to(device)

# PyTorch loss function expects float results of shape (batch_size, 1) instead of (batch_size,)
# So we add a new axis and convert them to floats
y_train = torch.from_numpy(planar_targets[:, np.newaxis]).float().to(device)

print(f"x_train: {x_train.shape}. y_train: {y_train.shape}")

# %% [markdown] slideshow={"slide_type": "slide"}
# In order to use [mini-batch SGD](principles.ipynb), data needs to be passed to the model as small, randomized batches during training. The Pytorch [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) class abstracts this complexity for us.

# %% slideshow={"slide_type": "-"}
# Load data as randomized batches for training
planar_dataloader = DataLoader(
    list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True
)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Model definition
#
# A PyTorch model is defined by combining elementary blocks, known as *modules*.
#
# Our neural network uses the following ones:
# - [Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html): an ordered container of modules.
# - [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html): a linear transformation of its entries, a.k.a. *dense* or *fully connected* layer.
# - [Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html) and [Sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html): the corresponding activation functions.

# %% slideshow={"slide_type": "slide"}
# Create a MultiLayer Perceptron with 2 inputs and 1 output
# You may change its internal architecture:
# for example, try adding one neuron on the hidden layer and check training results
planar_model = nn.Sequential(
    # Hidden layer with 2 inputs
    nn.Linear(in_features=2, out_features=hidden_layer_size),
    nn.Tanh(),
    # Output layer
    nn.Linear(in_features=hidden_layer_size, out_features=1),
    nn.Sigmoid(),
).to(device)
print(planar_model)

# Count the total number of trainable model parameters (weights)
print(f"Number of trainable parameters: {count_parameters(planar_model)}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Loss function
#
# For binary classification tasks, the standard choice is the [binary cross entropy loss](classification.ipynb), conveniently provided by a [PyTorch class](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html).
#
# For each sample of the batch, it will compare the output of the model (a value $\in [0,1]$ provided by the sigmoid function) with the expected binary value $\in \{0,1\}$.

# %%
# Binary cross entropy loss function
planar_loss_fn = nn.BCELoss()


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Model training
#
# The training algorithm is as follows:
# - On each iteration on the whole dataset (known as an *epoch*) and for each data batch inside an epoch, the model output is computed on the current batch.
# - This output is used alongside expected results by the loss function to obtain the mean loss for the current batch.
# - The gradient of the loss w.r.t. each model parameter is computed (backpropagation).
# - The model parameters are updated in the opposite direction of their gradient (one GD step).
#


# %% slideshow={"slide_type": "slide"}
def fit_planar(dataloader, model, loss_fn, epochs):
    """Main training code"""

    for _ in range(epochs):
        # Training algorithm for one data batch (i.e. one gradient descent step)
        for x_batch, y_batch in dataloader:
            # Forward pass: compute model output with current weights
            output = model(x_batch)

            # Compute loss (comparison between expected and actual results)
            loss = loss_fn(output, y_batch)

            # Reset the gradients to zero before running the backward pass
            # Avoids accumulating gradients between gradient descent steps
            model.zero_grad()

            # Backward pass (backprop): compute gradient of the loss w.r.t each model weight
            loss.backward()

            # Gradient descent step: update the weights in the opposite direction of their gradient
            # no_grad() avoids tracking operations history, which would be useless here
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad


# %% slideshow={"slide_type": "slide"}
# Fit model to planar data
fit_planar(
    dataloader=planar_dataloader,
    model=planar_model,
    loss_fn=planar_loss_fn,
    epochs=n_epochs,
)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Training results

# %% slideshow={"slide_type": "-"}
plot_decision_boundary(planar_model, planar_data, planar_targets)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Multiclass classification example

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Data loading and visualization
#
# We use the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, analogous to the famous MNIST handwritten digits dataset. It consists of:
# - a training set containing 60,000 28x28 grayscale images, each of them associated with a label (fashion category) from 10 classes;
# - a test set of 10,000 images with the same properties.
#
# A [PyTorch class](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) simplifies the loading process of this dataset.

# %% slideshow={"slide_type": "slide"}
fashion_train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

fashion_test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

# %% slideshow={"slide_type": "-"}
# Show info about the first training image
fashion_img, fashion_label = fashion_train_data[0]

print(f"First image: {fashion_img.shape}. Label: {fashion_label}")

# %% slideshow={"slide_type": "slide"}
# Show raw data for the first image
# Pixel values have already been normalized into the [0,1] range
print(fashion_img)

# %% slideshow={"slide_type": "slide"}
# Labels, i.e. fashion categories associated to images (one category per image)
fashion_labels = (
    "T-Shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)


# %% slideshow={"slide_type": "slide"}
plot_fashion_images(data=fashion_train_data, labels=fashion_labels, device=device)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Hyperparameters

# %%
# Try to change the learning rate to 1e-2 ans check training results
learning_rate = 1e-3
n_epochs = 10
batch_size = 64

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Data preparation
#
# As always, data will be passed to the model as small, randomized batches during training.

# %%
fashion_train_dataloader = DataLoader(fashion_train_data, batch_size=batch_size)
fashion_test_dataloader = DataLoader(fashion_test_data, batch_size=batch_size)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Model definition
#
# Most PyTorch models are defined as subclasses of the [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class. Their constructor creates the layer architecture and their `forward` method defines the forward pass of the model.
#
# In this model, we use the [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) module that transforms an input tensor of any shape into a vector (hence its name).


# %% slideshow={"slide_type": "slide"}
class NeuralNetwork(nn.Module):
    """Neural network for fashion articles classification"""

    def __init__(self):
        super().__init__()

        # Flatten the input image of shape (1, 28, 28) into a vector of shape (28*28,)
        self.flatten = nn.Flatten()

        # Define a sequential stack of linear layers and activation functions
        self.layer_stack = nn.Sequential(
            # First hidden layer with 784 inputs
            nn.Linear(in_features=28 * 28, out_features=64),
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


# %% slideshow={"slide_type": "slide"}
fashion_model = NeuralNetwork().to(device)
print(fashion_model)

# Try to guess the total number of parameters for this model before running this code!
print(f"Number of trainable parameters: {count_parameters(fashion_model)}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Loss function
#
# The standard choice for multiclass classification tasks is the [cross entropy loss](classification.ipynb) a.k.a. negative log-likelihood loss, provided by a PyTorch class aptly named [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
#
# > PyTorch also offers the [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) class implementing the negative log-likelihood loss. A key difference is that `CrossEntropyLoss` expects *logits*  (raw, unnormalized predictions) as inputs, and uses [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) to transform them into probabilities before computing its output. Using `CrossEntropyLoss` is equivalent to applying `LogSoftmax` followed by `NLLLoss` ([more details](https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81)).


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Softmax
#
# The softmax function turns a vector $\pmb{v} = \{v_1, v_2, \dots, v_K \} \in \mathbb{R}^K$ of raws values (called a *logits vector* when it's the output of a ML model) into a probability distribution. It is a multiclass generalization of the sigmoid function.
#
# $$\sigma(\pmb{v})_k = \frac{e^{v_k}}{\sum_{k=1}^K {e^{v_k}}}\;\;\;\;
# \sum_{k=1}^K \sigma(\pmb{v})_k = 1$$
#
# - $K$: number of labels.
# - $\pmb{v}$: logits vector, i.e. raw predictions for each class.
# - $\sigma(\pmb{v})_k \in [0,1]$: probability associated to label $k \in [1,K]$.


# %% slideshow={"slide_type": "slide"}
def softmax(x):
    """Softmax function"""

    return np.exp(x) / sum(np.exp(x))


# Raw values (logits)
raw_predictions = [3.0, 1.0, 0.2]

probas = softmax(raw_predictions)
print(probas)

# Sum of all probabilities is equal to 1
print(sum(probas))

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Optimization algorithm
#
# PyTorch provides out-of-the-box implementations for many gradient descent optimization algorithms ([Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html), [RMSProp](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html), etc).
#
# We'll stick with vanilla mini-batch SGD for now.


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Model training
#
# In order to obtain more details about the training process, we define a [fit]() function that encapsulates the training code and computes metrics.


# %% slideshow={"slide_type": "slide"}
# Fit model to fashion images
fashion_history = fit(
    dataloader=fashion_train_dataloader,
    model=fashion_model,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optim.SGD(fashion_model.parameters(), lr=learning_rate),
    epochs=n_epochs,
    device=device,
)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Training results

# %% slideshow={"slide_type": "-"}
plot_loss_acc(fashion_history)

# %% slideshow={"slide_type": "slide"}
plot_fashion_images(
    data=fashion_train_data, labels=fashion_labels, device=device, model=fashion_model
)

# %%
