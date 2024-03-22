# %%
"""
Training functions
"""

# %%
from tqdm import tqdm
import torch


# %%
def get_device(use_mps=True):
    """Return the available GPU/CPU device along with an information message"""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        message = f"Using CUDA GPU {torch.cuda.get_device_name(0)} :)"
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        message = "Using MPS GPU :)"
    else:
        device = torch.device("cpu")
        message = "No GPU found, using CPU instead :\\"

    return device, message


# %%
def count_parameters(model, trainable=True):
    """Return the total number of (trainable) parameters for a model"""

    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable
        else sum(p.numel() for p in model.parameters())
    )


# %%
def epoch_loop(dataloader, model, loss_fn, optimizer, device):
    """Training algorithm for one epoch"""

    # Total loss for the current epoch
    total_loss = 0

    # Number of correct predictions for the current epoch
    n_correct = 0

    # Load data as batches of associated inputs and targets
    for x_batch, y_batch in tqdm(dataloader, unit="batches", ncols=100, colour="blue"):
        # Load data and targets on device memory
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Forward pass
        output = model(x_batch)
        loss = loss_fn(output, y_batch)

        # Backward pass: backprop and GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Accumulate data for epoch metrics: loss and number of correct predictions
            total_loss += loss.item()
            n_correct += (model(x_batch).argmax(dim=1) == y_batch).float().sum().item()

    return total_loss, n_correct


# %%
def fit(dataloader, model, loss_fn, optimizer, epochs, device):
    """Main training code"""

    # Object storing training history
    history = {"loss": [], "acc": []}

    # Number of samples
    n_samples = len(dataloader.dataset)

    # Number of batches in an epoch (= n_samples / batch_size, rounded up)
    n_batches = len(dataloader)

    print(f"Training started! {n_samples} samples. {n_batches} batches per epoch")

    for epoch in range(epochs):
        # Train model for one epoch
        total_loss, n_correct = epoch_loop(
            dataloader, model, loss_fn, optimizer, device
        )

        # Compute epoch metrics
        epoch_loss = total_loss / n_batches
        epoch_acc = n_correct / n_samples

        print(
            f"Epoch [{(epoch + 1):3}/{epochs:3}] finished. Mean loss: {epoch_loss:.5f}. Accuracy: {epoch_acc * 100:.2f}%"
        )

        # Record epoch metrics for later plotting
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)

    print(f"Training complete! Total gradient descent steps: {epochs * n_batches}")

    return history
