import torch
import torch.nn.functional as F

def cosine_similarity_score(probs: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity-based uncertainty score.

    Parameters:
    probs (torch.Tensor): Tensor of shape (batch_size, num_classes), representing predicted probability distributions or embeddings.

    Returns:
    torch.Tensor: Tensor of shape (batch_size,), representing 1 - cosine similarity between each sample and the mean embedding vector.
    Higher values indicate more uncertainty.
    """
    if probs.ndim != 2:
        raise ValueError("Expected 2D tensor (batch_size, num_classes)")

    # Normalize each row to unit vector
    normed = F.normalize(probs, p=2, dim=1)

    # Compute mean embedding
    mean_embedding = normed.mean(dim=0, keepdim=True)
    mean_embedding = F.normalize(mean_embedding, p=2, dim=1)

    # Cosine similarity between each sample and mean
    similarity = torch.sum(normed * mean_embedding, dim=1)

    # Convert similarity to uncertainty: higher similarity → lower uncertainty
    uncertainty = 1 - similarity
    return uncertainty
