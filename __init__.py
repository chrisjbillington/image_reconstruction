def reduced_chi2(a, b, unc, mask):
    """Compute the sum squared error/uncertainty per degree of
    freedom between arrays a and b with given uncertainties and mask"""
    return ((a[mask].astype(float) - b[mask].astype(float))**2 /
             unc[mask].astype(float)**2).sum()/mask.sum()