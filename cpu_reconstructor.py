from __future__ import division, print_function
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from . import reduced_chi2


def pca(BT):
    """Principal component analysis on set of reference vectors that are rows
    of BT. Returns the mean vector, principal components as rows of a matrix
    XT, and variance explained by each, with the most important principal
    components first. If the number of elements in each vector is greater than
    the number of vectors, then n_elements principal components are returned.
    otherwise n_vectors - 1 are returned.
    """
    num_refs, n_pixels = BT.shape

    # Center the data about its mean
    mean_vector = BT.mean(axis=0)
    XT = BT - mean_vector

    X = XT.T

    if num_refs <= n_pixels:
        # Do PCA in the 'image basis' rather than the 'pixel basis' to avoid
        # constucting a potentially massive n_pixels x n_pixels matrix. Google
        # "PCA compact trick" for more info about this trick.

        # Compute the covariance matrix in the image basis, (X.T X)
        covariance_image_basis = np.dot(XT, X)

        # recover some memory:
        del XT

        # Diagonalise the covariance matrix:
        evals, evecs_image_basis = np.linalg.eigh(covariance_image_basis)

        # recover some memory:
        del covariance_image_basis

        # Discard the eigenvector with smallest eigenvalue, due to the
        # centering of the data, it is not orthogonal to the rest.
        evals = evals[1:]
        evecs_image_basis = evecs_image_basis[:, 1:]

        # Convert eigenvectors back into the pixel basis and normalise
        evecs_pixel_basis = np.dot(X, evecs_image_basis)

        # recover some memory:
        del X
        del evecs_image_basis

        evecs_pixel_basis /= np.linalg.norm(evecs_pixel_basis, axis=0)

    else:
        # Do PCA in the 'pixel basis', since it has a smaller number of
        # dimensions than the image basis:

        # Compute the covariance matrix in the pixel basis, (X X.T):
        covariance_pixel_basis = np.dot(X, XT)

        # recover some memory:
        del XT
        del X

        # Diagonalise the covariance matrix:
        evals, evecs_pixel_basis = np.linalg.eigh(covariance_pixel_basis)

        # recover some memory:
        del covariance_pixel_basis

    # Convert the eigenvectors to row vectors:
    principal_components = evecs_pixel_basis.T

    # By centering the data we've made the basis not linearly independent. The
    # first eigenvector (the one with the smallest eigenvalue - it will be
    # close to zero) is not orthogonal to the others. Discard it.

    # Reverse order of principal components and corresponding eigenvalues to
    # put most important ones (i.e. biggest eigenvalues) first:
    principal_components = principal_components[::-1]
    evals = evals[::-1]

    return mean_vector, principal_components, evals


class CPUReconstructor(object):
    def __init__(self, max_ref_images=None):
        self.max_ref_images = max_ref_images
        self.initialised = False
        
    def _init(self, ref_image):
        self.n_pixels = ref_image.size
        self.image_shape = ref_image.shape
        if self.max_ref_images is None:
            self.max_ref_images = int(np.sqrt(self.n_pixels))
        # Array of reference images as rows (hence equal to B.T). It's
        # initially full of zeros, but we exclude the uninitialised
        # parts of the array from being used as reference images.
        self.BT = np.zeros((self.max_ref_images, self.n_pixels))
        self.next_ref_image_index = 0
        self.initialised = True
        self.ref_image_hashes = []
        self.n_ref_images = 0

        # Cache the results of PCA:
        self.pca_results = None

        # Cache the some expensive-to-compute arrays:
        self.cached_arrays = None
        
    def add_ref_image(self, ref_image):
        """Add a reference image to the array of reference images used for
        reconstruction"""
        if not self.initialised:
            self._init(ref_image)
            
        # Hash the image to check for uniqueness:
        imhash = hash(ref_image.tobytes())
        if imhash in self.ref_image_hashes:
            # Ignore duplicate image
            return
        if self.n_ref_images < self.max_ref_images:
            self.ref_image_hashes.append(imhash)
        else:
            self.ref_image_hashes[self.next_ref_image_index] = imhash
        self.n_ref_images = len(self.ref_image_hashes)
        
        # Insert the new image into the correct row of BT:
        self.BT[self.next_ref_image_index] = ref_image.flatten().astype(float)
        # Move our index along by one for where the next reference image will go:
        self.next_ref_image_index += 1
        # Wrap around to overwrite oldest images:
        self.next_ref_image_index %= self.max_ref_images

        # Mark PCA as out of date
        self.pca_results = None

        # Mark cached arrays as out of date
        self.cached_arrays = None


    def add_ref_images(self, ref_images):
        """Convenience function to add many reference images"""
        for ref_image in ref_images:
            self.add_ref_image(ref_image)
            
    def pca_images(self):
        """Return mean_image, principal_component_images, variances, the same
        as update_pca() except reshaped to the image shape"""
        mean_vector, principal_components, variances = self.pca()
        shape = (len(principal_components),) + self.image_shape
        principal_component_images = principal_components.reshape(shape)
        mean_image = mean_vector.reshape(self.image_shape)
        return mean_image, principal_component_images, variances

    def pca(self):
        """Check if principal component analysis has been done on the current
        set of reference images and do it if it hasn't. Return mean_vector,
        principal_components, variances"""
        if not self.initialised and self.pca_results is None:
            msg = "No reference images added or previously computed PCA basis loaded"
            raise RuntimeError(msg)
        if self.pca_results is None:
            self.pca_results = pca(self.BT[:self.n_ref_images])
        return self.pca_results

    def save_pca(self, filepath):
        """Save cached PCA results to disk"""
        mean_vector, principal_component, evals = self.pca()
        image_shape = np.array(self.image_shape)
        with open(filepath, 'wb') as f:
            np.save(f, image_shape)
            np.save(f, mean_vector)
            np.save(f, principal_component)
            np.save(f, evals)

    def load_pca(self, filepath):
        """Restore saved PCA results from disk. Since you may load any
        previously computed PCA basis, this may or may not be consistent with
        any reference images you have added (and you may have not added any
        reference images at all). It is up to you to keep track of this: if
        you call reconstruct() with n_principal_components not None, the PCA
        basis will be used, otherwise the reference images will be used. If
        you add more reference images, the PCA basis will deleted and
        recomputed from the set of reference images. This could lead to subtle
        mistakes if you are not careful."""
        with open(filepath, 'rb') as f:
            image_shape = np.load(f, allow_pickle=False)
            mean_vector = np.load(f, allow_pickle=False)
            principal_components = np.load(f, allow_pickle=False)
            evals = np.load(f)

        image_shape = tuple(image_shape)
        if not self.initialised:
            self._init(np.empty(image_shape))
        elif self.image_shape != image_shape:
            msg = 'image shape does not match'
            raise ValueError(msg)
        self.pca_results = mean_vector, principal_components, evals
        # Mark cached arrays as out of date:
        self.cached_arrays = None

    def reconstruct(self, image, uncertainties=None, mask=None,
                    n_principal_components=None, return_coeffs=False):
        """Reconstruct image as a sum of reference images based on the
        weighted least squares solution in the region where mask=1. If
        uncertainties is None, all ones will be used. If mask is None, all
        True will be used. If n_principal_components is not None, the
        reconstruction will use the requested number of principal components
        of the reference images instead of the reference images directly. If
        return_coeffs is True, a list of coefficients for the linear sum is
        also returned. Note that since images are centred prior to PCA and
        reconstruction when using PCA, the difference between the mean image
        and the target image is what is reconstructed using a linear sum of
        principal components, so the reconstructed image is:

        recon_image = mean_image + \sum_i coeff_i * pca_basis_vector_i"""
        if not self.initialised and self.pca_results is None:
            msg = "No reference images added or previously computed PCA basis loaded"
            raise RuntimeError(msg)

        if uncertainties is None:
            # Reconstruction will be unweighted:
            uncertainties = np.ones(image.shape)
            
        if mask is None:
            # Reconstruction will be unmasked:
            mask = np.ones(image.shape, dtype=bool)
        else:
            # Convert to bool if not already:
            mask = mask.astype(bool)

        # Calculate weights, and convert and reshape arrays:
        W = (mask/uncertainties**2).astype(float).flatten()
        a = image.reshape(image.size, 1).astype(float)

        if n_principal_components is not None:
            # Center the image and use the principal components as reference
            # basis:
            mean_vector, principal_components, _ = self.pca()
            a -= mean_vector[:, None]
            BT = principal_components[:n_principal_components]
        else:
            BT = self.BT[:self.n_ref_images]

        # Now we solve the weighted least squares problem
        #     (B.T W B) x = (B.T W) a
        # for x and then reconstruct the image as:
        #     a_rec = B x
        
        
        cache_valid = False

        # Check if we've cached the arrays we need:
        if self.cached_arrays is not None:
            cached_n_pc, cached_W, cached_arrays = self.cached_arrays
            # If the weights and number of principal_components (or None)
            # are the same as when the arrays were cached, then use the cache:
            if cached_n_pc == n_principal_components and np.array_equal(cached_W, W):
                BTW, B, BTWB_LU_decomp = cached_arrays
                cache_valid = True

        if not cache_valid:
            # If there was nothing cached, or if the weights or number of
            # principal components didn't match the cache, compute the arrsys
            # from scratch:
            
            # Compute (B.T W) and B:
            BTW = BT * W
            B = BT.T

            # Compute the LHS of the linear system, (B.T W B)
            BTWB = np.dot(BTW, B)

            # LU factor the LHS of the linear system. We will solve the system
            # by LU decomposition and then calling scipy.linalg.lu_solve. This
            # is the same algorithm as calling np.linalg.solve(), but allows
            # us to cache the intermediate LU decomposition in case it can be
            # used again:
            BTWB_LU_decomp = lu_factor(BTWB)

            # Cache the arrays for next time, since they are expensive to
            # recompute. Save the mask and n_principal_components, since re-
            # using the cache is only valid if they are the same. The only downside
            # to this I think is extra memory consumption. B should be a view, so 
            # should not consume condiderable memory, but BTW is large.
            self.cached_arrays = n_principal_components, W, (BTW, B, BTWB_LU_decomp)
        
        # Compute the RHS of the linear system, (B.T W) a:
        BTWa = np.dot(BTW, a)

        # Solve the linear system:
        x = lu_solve(BTWB_LU_decomp, BTWa)

        # Do the reconstruction a_rec = B x:
        a_rec = np.dot(B, x)
        
        if n_principal_components is not None:
            # Add back on the mean vector:
            a_rec += mean_vector[:, None]

        reconstructed_image = a_rec.reshape(image.shape)

        # Compute reduced chi2 error statistic:
        rchi2_recon = reduced_chi2(image, reconstructed_image, uncertainties, mask)

        if return_coeffs:
            return reconstructed_image, rchi2_recon, x.flatten()
        else:
            return reconstructed_image, rchi2_recon
