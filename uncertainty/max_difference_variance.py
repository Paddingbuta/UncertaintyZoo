import torch

def max_difference_variance(probs: torch.Tensor) -> torch.Tensor:
    """
    Calculate the variance between the max probability and the rest of the class probabilities for each sample.

    Args:
        probs (torch.Tensor): Tensor of shape (batch_size, num_classes), each row is a probability distribution.

    Returns:
        torch.Tensor: Tensor of shape (batch_size,), where each element is the variance of the differences.
    """
    # Check for zero probabilities (e.g., after softmax malfunction or masking)
    if torch.any(probs.sum(dim=1) == 0):
        raise ValueError("Each sample's probability distribution must sum to a non-zero value.")

    max_probs, _ = torch.max(probs, dim=1, keepdim=True)  # shape: (batch_size, 1)
    differences = max_probs - probs  # shape: (batch_size, num_classes)
    variances = torch.var(differences, dim=1, unbiased=False)  # shape: (batch_size,)
    return variances
