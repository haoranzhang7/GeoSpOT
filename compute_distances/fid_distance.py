import torch
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        _model.fc = torch.nn.Identity()
        _model.eval().to(device)
    return _model

@torch.no_grad()
def get_features(images, batch_size=32):
    feats = []
    for i in range(0, len(images), batch_size):
        x = images[i:i+batch_size].to(device)
        x = 2 * x - 1  # [0,1] -> [-1,1] like Inception expects
        feats.append(_get_model()(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


def stats(features):
    return np.mean(features, axis=0), np.cov(features, rowvar=False)

def fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)


def calculate_fid(images1, images2, batch_size=32):
    f1 = get_features(images1, batch_size)
    f2 = get_features(images2, batch_size)

    mu1, s1 = stats(f1)
    mu2, s2 = stats(f2)

    return fid(mu1, s1, mu2, s2)