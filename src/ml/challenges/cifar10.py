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
# - Regarding data preparation, bitmap images should be normalized. You can find an example [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#load-and-normalize-cifar10).
# - You may train and evaluate a standard MultiLayer Perceptron, using [this chapter](../ann.ipynb) as a blueprint. Warning: you will have to adapt the inputs of your network to the color images of the CIFAR10 dataset.
# - **Bonus**: train a convolutional neural network using [this chapter](../cnn.ipynb) as a blueprint. After training, compare its performance with the MLP results.


# %%
# pylint: disable=undefined-variable

# Utility functions


def plot_loss_acc(history):
    """Plot training loss and accuracy. Takes a Keras-like History object as parameter"""

    loss_values = history["loss"]
    recorded_epochs = range(1, len(loss_values) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(recorded_epochs, loss_values, ".--", label="Training loss")
    ax1.set_ylabel("Loss")
    ax1.legend()

    acc_values = history["acc"]
    ax2.plot(recorded_epochs, acc_values, ".--", label="Training accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.legend()

    final_loss = loss_values[-1]
    final_acc = acc_values[-1]
    fig.suptitle(
        f"Training loss: {final_loss:.5f}. Training accuracy: {final_acc*100:.2f}%"
    )
    plt.show()


def count_parameters(model, trainable=True):
    """Return the total number of (trainable) parameters for a model"""

    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable
        else sum(p.numel() for p in model.parameters())
    )


def plot_images(data, labels, model=None):
    """Plot some images with either their true or predicted labels"""

    figure = plt.figure(figsize=(10, 6))
    cols, rows = 8, 4
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)

        # Title is either true or predicted label
        if model is None:
            title = labels[label]
        else:
            # Add a dimension (to match expected shape with batch size) and store image on device memory
            x_img = img[None, :].to(device)
            # Compute predicted label for image
            # Even if the model outputs unormalized logits, argmax gives the predicted label
            pred_label = model(x_img).argmax(dim=1).item()
            title = f"{labels[pred_label]}?"
        plt.title(title)

        plt.axis("off")
        img = img / 2 + 0.5  # unnormalize
        npimg = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
        plt.imshow(npimg, cmap="binary")
    plt.show()
