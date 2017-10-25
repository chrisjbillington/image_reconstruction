# image_reconstruction #

A module for reconstructing images as a linear sum of reference images based
on weighted least squares, optionally using a truncated basis of principal
components of the reference images rather than the reference images
themselves. Arrays of any dimension can be reconstructed, not only 2D images.

The use of weighted least squares allows uncertainties in the target image to
be taken into account as well as a region to be masked such that it can be
reconstructed based on the best fit in the unmasked region. This allows you to
mask a feature of an image and reconstruct what the reference images predict
the image would look like there in the absence of that feature.

The primary use of this for me is absorption imaging of cold atoms. Two images
are taken, one with atoms absorbing some of the imaging light, and one just
with the imaging light (the 'probe' or 'flat' image). The ratio between the
two is the quantity of interest, but because laser powers may change a bit or
things might be vibrating a bit, the two images do not perfectly correspond to
the same imaging light profile - it might be a bit different between the two,
even if they are taken within milliseconds of each other. So we can
reconstruct from the image with the atoms in it what imaging light profile, as
a linear sum of a number of previously acquired probe images, fits it best in
the region where the atoms are not present, and then assume the same linear
sum holds in the region where the atoms are present (which is masked out to
have zero weight in  the reconstruction). So we can reconstruct the 'best'
probe image from the atoms image (plus a mask to zero the weights of the fit
where atoms are present) and a set of probe images.

Moreover, the probe image reconstructed this way has much lower photon shot
noise, since it is a linear sum of other probe images and their noise cancels
somewhat. This can improve the signal-to-noise of the resulting images even if
the imaging has good laser and vibrational stability.

When the number of reference images becomes comparable to the number of pixels
in the image, there is a risk of overfitting - a reconstructed image is a sum
of so many reference images that there are enough degrees of freedom for the
weighted least squares fit to reproduce all the noise in the unmasked region -
likely to the detriment of the reconstruction in the masked region. When this
is a risk the basis can be truncated using principle component analysis, and
images reconstructed using a number of principal components much smaller than
the number of pixels, avoiding overfitting whilst using the basis that
describes most of the variation within the set of reference images.


### installation ###

To install, clone this repository somewhere in your PYTHONPATH or into the
working directory of your project.

### Usage ###

See test/test.py for examples.

There are two classes for image reconstruction - CPUReconstructor and
CUDAReconstructor. The latter does its calculations on a GPU using CUDA, if
available. This is much faster, and necessary for large numbers of images
since the bottleneck of the reconstruction algorithm runs in quadratic time
with respect to the number of images, but parallelises nicely.

The principal component analysis functionality is only implemented in the CPU
reconstructor, but there is no barrier to it being implemented using CUDA as
