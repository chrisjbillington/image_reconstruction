from __future__ import division, print_function
import numpy as np


class ImageAverager(object):
    """Class to make moving average of images as well as their variance"""
    def __init__(self, max_ref_images=None):
        self.n_ref_images = 0
        self.max_ref_images = max_ref_images
        self.next_ref_image_index = 0
        self.ref_image_hashes = []
        self.ref_images = []

    def add_ref_image(self, image):
        # Hash the image to check for uniqueness:
        imhash = hash(image.tobytes())
        if imhash in self.ref_image_hashes:
            # Ignore duplicate image
            return
        image = image.astype(float)
        if self.n_ref_images < self.max_ref_images:
            self.ref_image_hashes.append(imhash)
            self.ref_images.append(image)
        else:
            self.ref_image_hashes[self.next_ref_image_index] = imhash
            self.ref_images[self.next_ref_image_index] = image
        self.n_ref_images = len(self.ref_images)
        # Move our index along by one for where the next reference image will go:
        self.next_ref_image_index += 1
        # Wrap around to overwrite oldest images:
        self.next_ref_image_index %= self.max_ref_images

    def get_average(self):
        return np.mean(self.ref_images, axis=0)
        
    def get_variance(self):
        return np.var(self.ref_images, axis=0)
        
   
   
class ExpImageAverager(object):
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
            return self.average_image, self.variance_image
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

        