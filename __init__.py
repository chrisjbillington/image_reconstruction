import numpy as np


def reduced_chi2(a, b, unc, mask=None):
    """Compute the sum squared error/uncertainty per degree of
    freedom between arrays a and b with given uncertainties and mask"""
    if mask is None:
        # No masek
        mask = np.ones(a.shape, dtype=bool)
    else:
        # Convert to bool if not already:
        mask = mask.astype(bool)
    return ((a[mask].astype(float) - b[mask].astype(float))**2 /
             unc[mask].astype(float)**2).sum()/mask.sum()
