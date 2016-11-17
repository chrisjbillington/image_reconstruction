# image_reconstruction #

A module for reconstructing images as a linear sum of reference images based on weighted least squares.

### installation ###

To install, clone this repository somewhere in yout PYTHONPATH or into the working directory of your project.

### Usage ###

See test/test.py for an example.

There are two classes for image reconstruction - CPUReconstructor and CUDAReconstructor. The latter does its calculations on a GPU using CUDA, if available. This is much faster, and necessary for large numbers of images since the bottleneck of the reconstruction algorithm runs in quadratic time with respect to the number of images, but parallelises nicely.