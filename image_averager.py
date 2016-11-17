from __future__ import division, print_function
import numpy as np

class ImageAverager(object):
	"""Class to make exponential moving average of images as well as their variance"""
    def __init__(self, max_ref_images=None):
        self.average_image = None
        self.variance_image = None
        self.n_ref_images = 0
        self.max_ref_images = max_ref_images

    def update(self, image):
        image = image.astype(float)
        if self.n_ref_images < self.max_ref_images:
            self.n_ref_images += 1
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
