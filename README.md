# image_reconstruction #

A module for reconstructing images as a linear sum of reference images based
on weighted least squares, optionally using a truncated basis of principal
components of the reference images rather than the reference images
themselves. Arrays of any dimension can be reconstructed, not only 2D images.

The use of weighted least squares allows uncertainties in the target image to
be taken into account as well as a region to be masked from the reconstruction. 

### installation ###

To install, clone this repository somewhere in yout PYTHONPATH or into the
working directory of your project.

### Usage ###

See test/test.py for examples.

There are two classes for image reconstruction - CPUReconstructor and
CUDAReconstructor. The latter does its calculations on a GPU using CUDA, if
available. This is much faster, and necessary for large numbers of images
since the bottleneck of the reconstruction algorithm runs in quadratic time
with respect to the number of images, but parallelises nicely.

The principal component analysis functionality is only implenmented in the CPU
reconstructor, but there is no barrier to it being implemented using CUDA as
