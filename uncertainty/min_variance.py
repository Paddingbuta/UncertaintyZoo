import torch

def min_variance(probabilities: torch.Tensor) -> torch.Tensor:
    """
    Computes the minimum variance across class probabilities for each instance in the batch.
    Lower variance indicates more uncertainty in the prediction.

    Args:
        probabilities (torch.Tensor): A tensor of shape (batch_size, num_classes) representing predicted probabilities.

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) with the minimum variance scores.
    """
    # Handle edge case: all probabilities are zero
    row_sums = probabilities.sum(dim=1, keepdim=True)
    zero_mask = (row_sums == 0)

    # Normalize to make sure it's probability distribution
    normalized_probs = probabilities.clone()
    normalized_probs[~zero_mask] = normalized_probs[~zero_mask] / row_sums[~zero_mask]

    # Variance along classes
    variances = normalized_probs.var(dim=1, unbiased=False)

    # For all-zero rows, set variance to maximum (1.0) to reflect maximum uncertainty
    variances[zero_mask.squeeze()] = 1.0

    return variances
