import torch

def sample_variance(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute Sample Variance as an uncertainty measure.

    Args:
        probs (torch.Tensor): Tensor of shape (num_samples, batch_size, num_classes),
                              where num_samples is the number of stochastic forward passes,
                              and each [i] is a softmax probability output.

    Returns:
        torch.Tensor: Uncertainty score of shape (batch_size,), representing
                      the average variance across classes for each input.
    """
    # Calculate variance across the samples (dim=0)
    variance = torch.var(probs, dim=0)  # shape: (batch_size, num_classes)

    # Average the variance across the class dimension
    uncertainty = variance.mean(dim=1)  # shape: (batch_size,)

    return uncertainty
