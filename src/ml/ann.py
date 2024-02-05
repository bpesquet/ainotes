# %% [markdown] slideshow={"slide_type": "slide"}
# # Artificial neural networks

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "slide"}
import platform

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %% slideshow={"slide_type": "slide"}
# sklearn does not automatically import its subpackages
# https://stackoverflow.com/a/9049246/2380880
import sklearn
from sklearn.datasets import make_circles

import torch
from torch import nn, optim

# %% slideshow={"slide_type": "slide"}
# Print environment info
print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"PyTorch version: {torch.__version__}")


# PyTorch device configuration
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f"CUDA GPU {torch.cuda.get_device_name(0)} found :)")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Metal GPU found :)")
# else:
device = torch.device("cpu")
print("No available GPU, failing back to CPU :/")


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


def plot_2d_data(x, y):
    """Plot some 2D data"""

    plt.figure()
    plt.plot(x[y == 0, 0], x[y == 0, 1], "or", alpha=0.5, label=0)
    plt.plot(x[y == 1, 0], x[y == 1, 1], "ob", alpha=0.5, label=1)
    plt.legend()
    plt.show()


x_train, y_train = make_circles(n_samples=500, noise=0.1, factor=0.3)
plot_2d_data(x_train, y_train)

# Create PyTorch tensors from NumPy tensors
x_train = torch.from_numpy(np.float32(x_train)).to(device)
y_train = torch.from_numpy(np.float32(y_train[:, np.newaxis])).to(device)

print(f"x_train: {x_train.shape}. y_train: {y_train.shape}")


class MLPClassifier(nn.Module):
    """Define a simple neural network for baniry classification"""

    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """Define operations for forward pass"""

        return self.stack(x)


mlp_clf = MLPClassifier().to(device)
print(mlp_clf)

# Hyperparameters
learning_rate = 0.1
num_epochs = 100

loss_fn = nn.BCELoss()
optimizer = optim.SGD(mlp_clf.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    y_pred = mlp_clf(x_train)
    loss = loss_fn(y_pred, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.5f}")
