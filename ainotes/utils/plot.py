# %%
"""
Plotting functions
"""

# %%
import matplotlib.pyplot as plt
import torch
import numpy as np


# %%
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


# %%
def plot_fashion_images(data, labels, device, model=None):
    """Plot some images with their associated or predicted labels"""

    figure = plt.figure(figsize=(10, 6))
    cols, rows = 7, 4
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
        plt.imshow(img.cpu().detach().numpy().squeeze(), cmap="gray")
    plt.show()


# %%
def plot_cifar10_images(data, labels, device, model=None):
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
