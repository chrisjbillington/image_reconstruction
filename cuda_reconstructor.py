from __future__ import division, print_function
import numpy as np
import pycuda.autoinit
from pycuda.curandom import XORWOWRandomNumberGenerator
from pycuda import gpuarray

import skcuda
import skcuda.linalg
import skcuda.misc


class CUDAReconstructor(object):
    def __init__(self, max_ref_images=None):
        self.max_ref_images = max_ref_images
        self.initialised = False
        
    def _init(self, ref_image):
        skcuda.linalg.init()
        self.n_pixels = ref_image.size
        if self.max_ref_images is None:
            self.max_ref_images = int(np.sqrt(self.n_pixels))
        # GPU array of reference images as rows (hence equal to B.T).
        # It's initially full of random data.
        self.BT_gpu = XORWOWRandomNumberGenerator().gen_normal(
            (self.max_ref_images, self.n_pixels), float)
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
        
        # Send flattened, double precision reference image to the GPU:
        gpu_ref_image = gpuarray.to_gpu(ref_image.flatten().astype(float))
        # Compute 1D indices for the location in BT
        # where the new reference image will be inserted:
        start_index = self.n_pixels * self.next_ref_image_index
        stop_index = start_index + self.n_pixels
        insertion_indices = gpuarray.arange(start_index, stop_index, 1, dtype=int)
        # Insert the new image into the correct row of gpu_BT.
        skcuda.misc.set_by_index(self.BT_gpu, insertion_indices, gpu_ref_image)
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
        W = (mask/uncertainties**2).reshape(image.size, 1).astype(float)
        a = image.reshape(image.size, 1).astype(float)

        # Send image and weights to the GPU:
        a_gpu = gpuarray.to_gpu(a)
        W_gpu = gpuarray.to_gpu(W)
        
        # Now we solve the weighted least squares problem
        #     (B.T W B) x = (B.T W) a
        # for x and then reconstruct the image as:
        #     a_rec = B x
        
        # Compute (B.T W):
        BTW_gpu = skcuda.linalg.dot_diag(W_gpu, self.BT_gpu, trans='Y')
        
        # Compute the LHS of the linear system, (B.T W B)
        BTWB_gpu = skcuda.linalg.dot(BTW_gpu, self.BT_gpu, transb='T')
        
        # Compute the RHS of the linear system, (B.T W) a:
        BTWa_gpu = skcuda.linalg.dot(BTW_gpu, a_gpu)

        # Solve the linear system on the CPU (don't have the
        # right Nvidia libraries to do it on the GPU, it's
        # not computationally expensive anyway):
        BTWB = BTWB_gpu.get()
        BTWa = BTWa_gpu.get()
        x = np.linalg.solve(BTWB, BTWa)

        # Do the reconstruction a_rec = B x on the GPU:
        x_gpu = gpuarray.to_gpu(x)
        a_rec_gpu = skcuda.linalg.dot(self.BT_gpu, x_gpu, transa='T')
        a_rec = a_rec_gpu.get()
        reconstructed_image = a_rec.reshape(image.shape)

        # Compute reduced chi2 error statistic:
        chi2_gpu = gpuarray.sum(W_gpu*(a_rec_gpu - a_gpu)**2)
        rchi2 = chi2_gpu.get()/mask.sum()

        return reconstructed_image, rchi2