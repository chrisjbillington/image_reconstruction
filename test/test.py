from __future__ import division, print_function, absolute_import


import numpy as np
import matplotlib
# matplotlib.use('Agg')

import sys
# Add path to modules
sys.path.append('../..')

from image_reconstruction import reduced_chi2

def test(ReconstructorClass, max_ref_images=None):

    import h5py
    import os
    import matplotlib.pyplot as plt

    # Get the test data from file:
    with h5py.File("test_data.h5") as f:
        ref_probes = f['ref_probes'][:]
        atoms_images = f['atoms_images'][:]
        probe_images = f['probe_images'][:]
        ROI_x = f.attrs["ROI_x"]
        ROI_y = f.attrs["ROI_y"]
        ROI_w = f.attrs["ROI_w"]
        ROI_h = f.attrs["ROI_h"]

    reconstructor = ReconstructorClass(max_ref_images)
    reconstructor.add_ref_images(ref_probes)
    
    imshape = ref_probes[0].shape

    # Construct the mask array:
    mask = np.ones(imshape, dtype=bool)
    mask[ROI_y:ROI_y + ROI_h, ROI_x:ROI_x + ROI_w] = 0

    recon_probe_dummy, rchi2_recon = reconstructor.reconstruct(probe_images[0],
                                                         np.sqrt(probe_images[0]), mask)
    print('"Reconstruction" error of a reference image (should be zero) :', rchi2_recon)
    print()

    # Reconstruct the probe image corresponding to each absorption image:
    for i, raw_atoms in enumerate(atoms_images):
        raw_probe = probe_images[i]

        import time
        start_time = time.time()
        # We reconstruct twice - once with sqrt(raw_atoms) as the uncertainty, and then
        # The second time with sqrt(recon_probe) as the uncertainty, which should be more accurate.
        recon_probe_1, _ = reconstructor.reconstruct(raw_atoms, np.sqrt(raw_atoms), mask)
        recon_probe_2, _ = reconstructor.reconstruct(raw_atoms, np.sqrt(recon_probe_1), mask)
        
        # Ignore the reduced chi2 returned from the above reconstructions and compute it in
        # terms of shot noise from the  best reconstruction:
        
        rchi2_orig = reduced_chi2(raw_probe, raw_atoms, np.sqrt(recon_probe_2), mask)
        rchi2_recon_1 = reduced_chi2(recon_probe_1, raw_atoms, np.sqrt(recon_probe_2), mask)
        rchi2_recon_2 = reduced_chi2(recon_probe_2, raw_atoms, np.sqrt(recon_probe_2), mask)
        print('         rchi2 orig:', rchi2_orig)
        print('      rchi2 recon_1:', rchi2_recon_1)
        print('      rchi2 recon_2:', rchi2_recon_2)
        print('         time taken:', time.time() - start_time, 'sec')
        print()

        # Compare optical density images (approximate for sake of example - no background subtraction or anything):
        raw_OD = -np.log(raw_atoms/raw_probe)
        reconstructed_OD = -np.log(raw_atoms/recon_probe_2)

        outdir = 'test_{}'.format(ReconstructorClass.__name__)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        fname_prefix = os.path.join(outdir, '%02d'%i)
        # Save resulting images:
        cmap='gray'
        plt.imsave(fname_prefix + "_0_raw_probe.png", raw_probe, vmin=0, vmax=raw_atoms.max(), cmap=cmap)
        plt.imsave(fname_prefix + "_1_raw_atoms.png", raw_atoms, vmin=0, vmax=raw_atoms.max(), cmap=cmap)
        plt.imsave(fname_prefix + "_2_recon_probe.png", recon_probe_2, vmin=0, vmax=raw_atoms.max(), cmap=cmap)
        plt.imsave(fname_prefix + "_5_raw_OD.png", raw_OD, vmin=0, vmax=reconstructed_OD.max(), cmap=cmap)
        plt.imsave(fname_prefix + "_5_recon_OD.png", reconstructed_OD, vmin=0, vmax=reconstructed_OD.max(), cmap=cmap)

        
def test_cuda():
    print("Testing CUDA reconstruction with ~500 reference images")
    # Note that we only have 23 reference images, so the others will be random noise.
    # This won't affect the reconstruction (much), but gives us a good estimate of
    # how long reconstruction would take if we did have ~500 reference images. 
    from image_reconstruction.cuda_reconstructor import CUDAReconstructor
    test(CUDAReconstructor)
    
    
def test_cpu():
    print("Testing CPU reconstruction with 50 reference images")
    # Note that we only have 23 reference images, so the others will be random noise.
    # This won't affect the reconstruction (much), but gives us a good estimate of
    # how long reconstruction would take if we did have ~50 reference images. 
    from image_reconstruction.cpu_reconstructor import CPUReconstructor
    test(CPUReconstructor, 50)
    
def test_pca():
    print('testing pca of 23 reference images')
    import h5py
    from image_reconstruction.cpu_reconstructor import CPUReconstructor
    reconstructor = CPUReconstructor(50)

    with h5py.File("test_data.h5") as f:
        ref_probes = f['ref_probes'][:]

    reconstructor.add_ref_images(ref_probes)
    mean_image, principal_components, evals = reconstructor.pca_images()

    outdir = 'pca_basis'
    import os
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    import matplotlib.pyplot as plt

    plt.plot(evals/mean_image.sum(), 'o-')
    plt.xlabel('principal component')
    plt.ylabel('variance explained in units of shot noise')
    plt.grid(True)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=21)
    plt.savefig('pca.png')

    for i, image in enumerate([mean_image] + list(principal_components)):
        fname_prefix = os.path.join(outdir, '%02d'%i)
        cmap='gray'
        plt.imsave(fname_prefix + ".png", image, cmap=cmap)

    

if __name__ == "__main__":
    test_cpu()
    test_pca()
    test_cuda()
    