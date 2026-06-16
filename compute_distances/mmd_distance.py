import torch
from torch import nn

def gaussian_kernel(X, Y, sigma=1.0):
    """
    Returns kernel matrix K where:
    K_ij = exp(-||x_i - y_j||^2 / (2 sigma^2))
    """
    dist2 = torch.cdist(X, Y) ** 2
    return torch.exp(-dist2 / (2 * sigma ** 2))

def linear_kernel(X, Y):
    """
    K_ij = x_i · y_j
    """
    return X @ Y.T

def mmd(X, Y, kernel_fn, **kernel_kwargs):
    """
    Computes the MMD distance between two sets of samples X and Y using the specified kernel function.
    """
    Z = torch.cat([X, Y], dim=0)

    # full kernel matrix
    K = kernel_fn(Z, Z, **kernel_kwargs)

    n = X.shape[0]
    
    K_xx = K[:n, :n]
    K_yy = K[n:, n:]
    K_xy = K[:n, n:]

    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


def median_heuristic_sigma(X, Y, max_samples=2000):
    Z = torch.cat([X, Y], dim=0)
    if len(Z) > max_samples:
        Z = Z[torch.randperm(len(Z))[:max_samples]]
    dists = torch.cdist(Z, Z)
    mask = ~torch.eye(len(Z), dtype=torch.bool, device=Z.device)
    return dists[mask].median().item()


def calculate_mmd(X, Y, kernel="multiscale", sigma=None, scales=(0.1, 0.5, 1.0, 2.0, 5.0)):
    """
    kernel="multiscale": sum of Gaussian MMDs at sigma * scales (recommended).
    kernel="gaussian":   single Gaussian; sigma defaults to median heuristic.
    kernel="linear":     linear kernel.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)

    if kernel == "linear":
        return mmd(X, Y, linear_kernel)

    base_sigma = sigma if sigma is not None else median_heuristic_sigma(X, Y)

    if kernel == "gaussian":
        return mmd(X, Y, gaussian_kernel, sigma=base_sigma)

    if kernel == "multiscale":
        return sum(mmd(X, Y, gaussian_kernel, sigma=base_sigma * s) for s in scales)

    raise ValueError(f"Unknown kernel: {kernel}")
