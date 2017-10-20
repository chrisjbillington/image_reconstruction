from __future__ import division, print_function
import numpy as np

from . import reduced_chi2

class CPUReconstructor(object):
    def __init__(self, max_ref_images=None):
        self.max_ref_images = max_ref_images
        self.initialised = False
        
    def _init(self, ref_image):
        self.n_pixels = ref_image.size
        if self.max_ref_images is None:
            self.max_ref_images = int(np.sqrt(self.n_pixels))
        # Array of reference images as rows (hence equal to B.T).
        # It's initially full of random data.
        self.BT = np.random.randn(self.max_ref_images, self.n_pixels)
        self.next_ref_image_index = 0
        self.initialised = True
        self.ref_image_hashes = []
        self.n_ref_images = 0
        
    def add_ref_image(self, ref_image):
        """Add a reference image to the array of reference images used for reconstruction"""
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
        

    def add_ref_images(self, ref_images):
        """Convenience function to add many reference images"""
        for ref_image in ref_images:
            self.add_ref_image(ref_image)
            
    def reconstruct(self, image, uncertainties, mask):
        """Reconstruct image as a sum of reference images based on the
        weighted least squares solution in the region where mask=1"""
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