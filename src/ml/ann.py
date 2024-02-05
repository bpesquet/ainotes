# %% [markdown] slideshow={"slide_type": "slide"}
# # Artificial neural networks

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "slide"}
import platform

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


# %% slideshow={"slide_type": "slide"}
# sklearn does not automatically import its subpackages
# https://stackoverflow.com/a/9049246/2380880
import sklearn
from sklearn.datasets import make_circles

import torch
from torch import nn

# %% slideshow={"slide_type": "slide"}
# Print environment info
print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"PyTorch version: {torch.__version__}")


# PyTorch device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA GPU {torch.cuda.get_device_name(0)} found :)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Metal GPU found :)")
else:
    device = torch.device("cpu")
    print("No GPU found, failing back to CPU :/")


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


# %% slideshow={"slide_type": "slide"}
# Utility functions


def plot_dataset(x, y):
    """Plot 2D data"""

    plt.figure()
    plt.plot(x[y == 0, 0], x[y == 0, 1], "or", label=0)
    plt.plot(x[y == 1, 0], x[y == 1, 1], "ob", label=1)
    plt.legend()
    plt.show()


def plot_decision_boundary(model, x, y, figure=None):
    """Plot a decision boundary"""

    if figure is None:  # If no figure is given, create a new one
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


def plot_loss_acc(training_history):
    """Plot training loss and accuracy. Takes a Keras-like History object as parameter"""

    loss_values = training_history["loss"]
    recorded_epochs = range(1, len(loss_values) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(recorded_epochs, loss_values, ".--", label="Training loss")
    ax1.set_ylabel("Loss")
    ax1.legend()

    acc_values = training_history["acc"]
    ax2.plot(recorded_epochs, acc_values, ".--", label="Training accuracy")
    ax2.set_xlabel("Recorded epochs")
    ax2.set_ylabel("Accuracy")
    plt.legend()

    final_loss = loss_values[-1]
    final_acc = acc_values[-1]
    fig.suptitle(f"Training loss: {final_loss:.5f}. Training accuracy: {final_acc:.4f}")
    plt.show()


# Generate 2D data (a large circle containing a smaller circle)
inputs, targets = make_circles(n_samples=500, noise=0.1, factor=0.3)
print(f"inputs: {inputs.shape}. targets: {targets.shape}")


plot_dataset(inputs, targets)

# Create PyTorch tensors from NumPy tensors

x_train = torch.from_numpy(inputs).float().to(device)

# PyTorch loss function expects float results of shape (batch_size, 1) instead of (batch_size,)
# So we add a new axis and convert them to floats
y_train = torch.from_numpy(targets[:, np.newaxis]).float().to(device)

print(f"x_train: {x_train.shape}. y_train: {y_train.shape}")

# Hyperparameters
learning_rate = 0.1
num_epochs = 5000
epoch_print_interval = 100


# Create a MultiLayer Perceptron with 2 inputs and 1 output
# (you may change the number of hidden neurons and layers)
mlp_clf = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1), nn.Sigmoid()).to(
    device
)
print(mlp_clf)

# Binary Cross Entropy loss function
loss_fn = nn.BCELoss()

# Object storing training history
history = {"loss": [], "acc": []}

# Training loop
for epoch in range(num_epochs):
    # Reset number of correct predictions for the current epoch
    nb_correct = 0

    # Forward pass: compute model output with current weights
    y_pred = mlp_clf(x_train)

    # Compute epoch loss (comparison between expected and actual results)
    loss = loss_fn(y_pred, y_train)

    # Zero the gradients before running the backward pass
    # Avoids accumulating gradients erroneously
    mlp_clf.zero_grad()

    # Backward pass (backprop): compute gradient of the loss w.r.t each model weight
    loss.backward()

    # Gradient descent step: update the weights in the opposite direction of their gradient
    # no_grad() avoids tracking operations history, which would be useless here
    with torch.no_grad():
        for param in mlp_clf.parameters():
            param -= learning_rate * param.grad

    # Compute epoch metrics, using a test to reduce verbosity
    if epoch == 0 or (epoch + 1) % epoch_print_interval == 0:
        # Compute epoch metrics
        epoch_loss = loss.item()
        nb_correct += (torch.round(y_pred) == y_train).float().sum().item()
        epoch_acc = nb_correct / len(y_train)

        # Record epoch metrics for later plotting
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)

        print(
            f"Epoch [{(epoch + 1):4}/{num_epochs:4}]. Loss: {epoch_loss:.5f}. Accuracy: {epoch_acc:.4f}"
        )


plot_decision_boundary(mlp_clf, inputs, targets)
plot_loss_acc(history)
