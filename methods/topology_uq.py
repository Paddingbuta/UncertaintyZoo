import torch
import torch.nn.functional as F
from typing import Dict, Optional, Literal

# Optional TDA stack (giotto-tda). If unavailable, we skip topo-entropy gracefully.
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams.features import PersistenceEntropy
    _HAS_TDA = True
except Exception:
    _HAS_TDA = False


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize vectors along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _trajectory_complexity(h: torch.Tensor, normalize_emb: bool = True) -> float:
    """
    Compute mean adjacent Euclidean distance over a trajectory h_t.
    h: [T, D]
    """
    if normalize_emb:
        h = _l2_normalize(h)
    diffs = h[1:] - h[:-1]                  # [T-1, D]
    dists = torch.norm(diffs, dim=-1)       # [T-1]
    return dists.mean().item() if dists.numel() > 0 else 0.0


def _discrete_curvature(h: torch.Tensor, normalize_emb: bool = True) -> float:
    """
    Curvature proxy via consecutive displacement angles:
      v_t = h_{t+1} - h_t
      kappa_t = 1 - cos_angle(v_{t-1}, v_t)
    Returns mean curvature in [0, 2].
    """
    if h.size(0) < 3:
        return 0.0
    if normalize_emb:
        h = _l2_normalize(h)
    v = h[1:] - h[:-1]                       # [T-1, D]
    v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
    cosines = (v_norm[1:] * v_norm[:-1]).sum(dim=-1)   # [T-2]
    cosines = torch.clamp(cosines, -1.0, 1.0)
    kappa = 1.0 - cosines                    # [T-2]
    return kappa.mean().item() if kappa.numel() > 0 else 0.0


def _topological_entropy(h: torch.Tensor, max_dim: int = 1) -> Optional[float]:
    """
    Persistent-entropy over a point cloud using Vietoris–Rips persistence.
    Requires giotto-tda. If unavailable, returns None.
    h: [T, D] (CPU numpy required by gtda)
    """
    if not _HAS_TDA or h.size(0) < 3:
        return None
    import numpy as np
    X = h.detach().cpu().numpy()[None, ...]  # shape: (1, T, D)
    vr = VietorisRipsPersistence(homology_dimensions=list(range(max_dim + 1)))
    diagrams = vr.fit_transform(X)           # (1, n_points, 3): [birth, death, dim]
    pe = PersistenceEntropy()
    ent = pe.fit_transform(diagrams)         # (1, )
    return float(ent[0])


def _final_hidden_trajectory_generative(model, prompt: str, max_new_tokens: int,
                                        temperature: float, top_p: float, top_k: int,
                                        repetition_penalty: float) -> torch.Tensor:
    """
    For causal LMs: generate tokens, then run one forward pass with
    output_hidden_states=True on the full sequence to extract a trajectory
    h_t defined as the final-layer hidden state at each generated token position.
    Returns h: [T, D] on CPU.
    """
    device = next(model.parameters()).device
    tok = model.tokenizer
    enc = tok(prompt, return_tensors="pt").to(device)
    gen_out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        return_dict_in_generate=True,
    )
    # Full generated sequence (prompt + generated)
    full_ids = gen_out.sequences
    # One forward with hidden states to get per-position embeddings
    with torch.no_grad():
        out_h = model(full_ids, output_hidden_states=True, use_cache=False)
    hidden_states = out_h.hidden_states[-1]  # [1, seq_len, hidden]
    prompt_len = enc["input_ids"].shape[1]
    gen_h = hidden_states[:, prompt_len:, :] # [1, T, D]
    return gen_h.squeeze(0).detach().cpu()   # [T, D]


def _trajectory_discriminative(model, text: str,
                               path: Literal["depth", "tokens"] = "depth",
                               max_length: int = 512) -> torch.Tensor:
    """
    Build a trajectory for encoder models:
      - depth: across layers using [CLS] (or first token) embedding
      - tokens: across positions using final-layer embeddings
    Returns h: [T, D] on CPU.
    """
    tok = model.tokenizer
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states  # list[L]: [1, S, D]
    if path == "depth":
        # use first token across layers
        traj = [h[:, 0, :] for h in hs]                 # L * [1, D]
        h = torch.cat(traj, dim=0)                      # [L, D]
    else:
        # use final layer across tokens
        last = hs[-1]                                   # [1, S, D]
        h = last.squeeze(0)                             # [S, D]
    return h.detach().cpu()


def topology_uq(
    model,
    text: str,
    model_type: Literal["generative", "discriminative"] = "generative",
    # Generative sampling params (only used when model_type == "generative")
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    # Discriminative path: "depth" or "tokens"
    disc_path: Literal["depth", "tokens"] = "depth",
    # Geometry controls
    normalize_emb: bool = True,
    compute_curvature: bool = True,
    compute_topo_entropy: bool = False,
    topo_max_dim: int = 1,
    # Score mixing
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Dict[str, float]:
    """
    Topology-Based Uncertainty (TopologyUQ)
    ---------------------------------------
    Quantifies reasoning instability via the geometry/topology of a hidden-state
    trajectory h_t in representation space.

    Construction of trajectory h_t:
      - Generative: final-layer hidden of each generated token position.
      - Discriminative:
          * "depth": [CLS]-like vector across encoder layers.
          * "tokens": final-layer vectors across token positions.

    Metrics:
      * Trajectory complexity (TC): mean adjacent Euclidean distance.
      * Discrete curvature: mean 1 - cos angle between consecutive displacements.
      * (Optional) Topological entropy via persistent homology (giotto-tda).

    Final score:
        TopologyUQ(x) = alpha * TC(x) + beta * TopoEntropy(x)

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Causal LM (generative) or encoder classifier (discriminative).
        Must have `model.tokenizer` attached.
    text : str
        Prompt (generative) or input text (discriminative).
    model_type : {"generative", "discriminative"}
        Which branch to use.
    max_new_tokens, temperature, top_p, top_k, repetition_penalty : ...
        Sampling parameters for generative trajectory extraction.
    disc_path : {"depth", "tokens"}
        Trajectory definition for discriminative models.
    normalize_emb : bool
        L2-normalize embeddings before geometry metrics.
    compute_curvature : bool
        Also report curvature proxy.
    compute_topo_entropy : bool
        If True (and giotto-tda is installed), compute persistent-entropy.
    topo_max_dim : int
        Max homology dimension for Vietoris–Rips (default 1).
    alpha, beta : float
        Mixing coefficients for the final score.

    Returns
    -------
    dict
        {
          "tc": float,                    # trajectory complexity
          "curvature": float,             # optional, 0 if disabled
          "topo_entropy": float or 0.0,   # optional or 0.0 if not computed
          "score": float                  # alpha*tc + beta*topo_entropy
        }
    """
    # ---- Build trajectory h_t ----
    if model_type == "generative":
        h = _final_hidden_trajectory_generative(
            model, text, max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )  # [T, D]
    elif model_type == "discriminative":
        h = _trajectory_discriminative(model, text, path=disc_path)  # [T, D]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Edge case: too short trajectory
    if h.size(0) < 2:
        tc = 0.0
        curv = 0.0
        topo_ent = 0.0
        return {"tc": tc, "curvature": curv, "topo_entropy": topo_ent, "score": alpha * tc + beta * topo_ent}

    # ---- Geometry metrics ----
    tc = _trajectory_complexity(h, normalize_emb=normalize_emb)
    curv = _discrete_curvature(h, normalize_emb=normalize_emb) if compute_curvature else 0.0

    # ---- Optional topological entropy ----
    topo_ent = 0.0
    if compute_topo_entropy:
        te = _topological_entropy(h, max_dim=topo_max_dim)
        topo_ent = float(te) if te is not None else 0.0

    score = alpha * tc + beta * topo_ent
    return {"tc": float(tc), "curvature": float(curv), "topo_entropy": float(topo_ent), "score": float(score)}
