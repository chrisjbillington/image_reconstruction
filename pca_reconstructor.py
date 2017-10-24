from __future__ import division, print_function
import numpy as np

from . import reduced_chi2
from .cpu_reconstructor import CPUReconstructor


def pca(BT):
    """Principal component analysis on set of reference
    vectors that are rows of BT. Returns the mean vector,
    principal components
    as columns of a matrix,
    and variance explained by each (normalised by the weights), with
    the most important principal components first.
    """
    # get dimensions
    num_refs, n_pixels = BT.shape

    if W is None:
        W = np.ones(n_pixels)

    # center data
    mean_vector = BT.mean(axis=0)
    XT = BT - mean_vector

    X = XT.T

    if num_refs < n_pixels:
        # Do PCA in the 'image basis' rather than the 'pixel basis' to avoid
        # constucting a potentially massive n_pixels x n_pixels matrix.
        # Google "PCA compact trick" for more info about this trick.

        # Compute (B.T W):
        XTW = XT * W
        
        # Compute the weighted covariance matrix in the image basis, (X.T W X)
        XTWX = np.dot(XTW, X)

        # Diagonalise the weighted covariance matrix:
        evals, evecs = np.linalg.eigh(XTWX)

        # Convert eigenvectors back into the pixel basis by summing over images.
        # The division by the square root of the eigenvalues is normalisation:
        evecs = np.dot(X, evecs).T / np.sqrt(evals)[:, None]

    # return the projection matrix, the variance and the mean
    return V,S,mean_X


class PCAReconstructor(CPUReconstructor):
    def pca(uncertainties, mask):
        """Perform weighted principal component analysis on the current
        set of reference images using the weights mask/uncertainties**2"""

            
    def reconstruct(self, image, uncertainties, mask, n_principal_components=None):
        """Reconstruct image as a linear sum of basis images based on the
        weighted least squares solution in the region where mask=1. The basis used is 
        the first n_principal_components principal components determined using weighted principal
        component analysis"""
        if not self.initialised:
            raise RuntimeError("No reference images added!")

        # Calculate weights, and convert and reshape arrays:
        W = (mask/uncertainties**2).astype(float).flatten()
        a = image.reshape(image.size, 1).astype(float)

        # Now we solve the weighted least squares problem
        #     (B.T W B) x = (B.T W) a
        # for x and then reconstruct the image as:
        #     a_rec = B x
        
        # Compute (B.T W):
        BTW = self.BT * W
        
        # Compute the LHS of the linear system, (B.T W B)
        B = self.BT.T
        BTWB = np.dot(BTW, B)
        
        # Compute the RHS of the linear system, (B.T W) a:
        BTWa = np.dot(BTW, a)

        # Solve the linear system:
        x = np.linalg.solve(BTWB, BTWa)

        # Do the reconstruction a_rec = B x:
        a_rec = np.dot(B, x)
        reconstructed_image = a_rec.reshape(image.shape)

        # Compute reduced chi2 error statistic:
        rchi2_recon = reduced_chi2(image, reconstructed_image, uncertainties, mask)

        return reconstructed_image, rchi2_recon