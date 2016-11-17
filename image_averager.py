from __future__ import division, print_function
import numpy as np

class ImageAverager(object):
    """Class to make exponential moving average of images as well as their variance"""
    def __init__(self, max_ref_images=None):
        self.average_image = None
        self.variance_image = None
        self.n_ref_images = 0
        self.max_ref_images = max_ref_images
        self.ref_image_hashes = []

    def update(self, image):
    
        # Hash the image to check for uniqueness:
        imhash = hash(image.data[:])
        if imhash in self.ref_image_hashes:
            # Ignore duplicate image
            return
        if self.n_ref_images < self.max_ref_images:
            self.ref_image_hashes.append(imhash)
        else:
            self.ref_image_hashes[self.next_ref_image_index] = imhash
        self.n_ref_images = len(self.ref_image_hashes)
        
        image = image.astype(float)
        k = 1/self.n_ref_images
        if self.average_image is None:
            self.average_image = image
            self.variance_image = np.zeros(image.shape, dtype=float)
        else:
            if image.shape != self.average_image.shape:
                raise ValueError("Cannot use images of different shapes/sizes")
            self.variance_image = (1 - k) * (self.variance_image + k * (image - self.average_image)**2)
            self.average_image = (1 - k) * self.average_image + k * image
        return self.average_image, self.variance_image