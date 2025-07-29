import torch

def class_probability_variance(probabilities: torch.Tensor) -> torch.Tensor:
    """
    Computes the Class Probability Variance (CPV) for a batch of predictions.

    Args:
        probabilities (torch.Tensor): Tensor of shape (T, B, C), where:
            T = number of stochastic forward passes (e.g., MC samples),
            B = batch size,
            C = number of classes.
    
    Returns:
        torch.Tensor: Tensor of shape (B,), CPV score for each instance in the batch.
    """
    if probabilities.dim() != 3:
        raise ValueError("Input tensor must be of shape (T, B, C)")
    
    # Compute variance across T samples for each class
    var_across_T = torch.var(probabilities, dim=0, unbiased=False)  # shape: (B, C)
    
    # Sum variances across classes
    cpv_score = var_across_T.sum(dim=1)  # shape: (B,)
    
    return cpv_score
