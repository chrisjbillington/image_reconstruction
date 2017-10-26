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
    with h5py.File("test_data.h5", 'r') as f:
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
    print("Testing CPU reconstruction with 23 reference images")
    from image_reconstruction.cpu_reconstructor import CPUReconstructor
    test(CPUReconstructor, 23)
    
def test_pca_basis():
    print('testing pca of 23 reference images')
    import h5py
    from image_reconstruction.cpu_reconstructor import CPUReconstructor
    # max 50 images, but there are actually only 23:
    reconstructor = CPUReconstructor(50) 

    with h5py.File("test_data.h5") as f:
        ref_probes = f['ref_probes'][:]

    reconstructor.add_ref_images(ref_probes)
    mean_image, principal_components, evals = reconstructor.pca_images()

    # Test saving PCA
    orig_pca_results = reconstructor.pca_results
    reconstructor.save_pca('pca')
    reconstructor.pca_results = None
    reconstructor.load_pca('pca')
    for arr1, arr2 in zip(orig_pca_results, reconstructor.pca_results):
        assert np.array_equal(arr1, arr2)

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
    plt.clf()

    for i, image in enumerate([mean_image] + list(principal_components)):
        fname_prefix = os.path.join(outdir, '%02d'%i)
        cmap='gray'
        plt.imsave(fname_prefix + ".png", image, cmap=cmap)


def test_1d_pca():

    import h5py
    import matplotlib.pyplot as plt
    from image_reconstruction.cpu_reconstructor import CPUReconstructor

    with h5py.File('test_1d_data.h5', 'r') as f:
        data = f['images'][:]

    # We have 19 29 x 490 images and are interested in reconstructing each
    # 29-pixel vertical slice of each image using a basis based on the same
    # slice and four nearest slices from the same image and 18 other images.

    def get_reference_slices(image_index, x_index):
        # Get the reference images to be used for reconstructing a particular slice
        start_index = x_index - 2
        stop_index = x_index + 3
        # Where, relative to the start index, is the slice we are reconstructing?
        offset = 2
        while start_index < 0:
            start_index += 1
            stop_index += 1
            offset -= 1
        while stop_index >= 490:
            start_index -= 1
            stop_index -= 1
            offset += 1

        # The slices, shape (19, 29, 5) 
        slices = data[:, :, start_index:stop_index]

        # Transpose to get the vertical dimension first. Shape (29, 19, 5)
        slices = slices.transpose((1, 0, 2))

        # Flatten the last two dimensions so we have a (29, 95) array:
        slices = slices.reshape(29, 5*19)

        # Transpose again so we have each realisation as a row. Shape (95, 29)
        slices = slices.transpose()

        # Delete the row corresponding to the slice we are going to reconstruct.
        # Resulting shape: (94, 29):
        slices = np.delete(slices, image_index*5 + offset, axis=0)

        # Verify for sure that the slice we're reconstructing is not in there:
        for i in range(94):
            assert not np.array_equal(slices[i], data[image_index, :, x_index])

        return slices

    # Let's reconstruct all the ODS in realisation 0
    image_index = 0
    image = data[0, :, :].copy()
    mean_image = np.zeros(image.shape)

    N = 5
    # Reconstruct slice by slice:
    reconstructed_image = np.zeros(image.shape)
    for x_index in range(490):
        print(x_index)
        target_slice = image[:, x_index]
        reference_slices = get_reference_slices(image_index, x_index)

        # Make a reconstructor with these realisations as reference images:
        reconstructor = CPUReconstructor(94)
        reconstructor.add_ref_images(reference_slices)

        reconstructed_slice, rchi2 = reconstructor.reconstruct(target_slice, n_principal_components=N)

        mean_slice, _, _ = reconstructor.pca_images()
        reconstructed_image[:, x_index] = reconstructed_slice
        mean_image[:, x_index] = mean_slice

    # reconstructed_image -= mean_image
    # image -= mean_image

    both_images = np.concatenate((image, reconstructed_image, image - reconstructed_image), axis=0)
    plt.imsave('1d_reconstruction.png', both_images)

    all_images = np.concatenate(data, axis=0)
    plt.imsave('1d_all_orig.png', all_images)

    plt.imsave('mean_image.png', mean_image)

    column_density_orig = image.sum(axis=0)
    plt.plot(column_density_orig, linewidth=1.0)
    
    column_density_recon = reconstructed_image.sum(axis=0)
    plt.plot(column_density_recon, label=str(N), linewidth=1.0)

    resid = column_density_recon - column_density_orig
    stderr = resid.std() / np.sqrt(len(resid))
    resid_mean_on_stderr = resid.mean()/stderr
    plt.plot(resid, label=str(N) + ' residuals\n($\mu$= %.01f stderrs)' % (resid_mean_on_stderr), linewidth=1)

    # mean_col_density = mean_image.sum(axis=0)
    # plt.plot(mean_col_density, label='mean')

    plt.legend()
    plt.grid(True)
    plt.savefig('1d_column_density.png')
    # plt.show()


if __name__ == "__main__":
    test_cpu()
    test_pca_basis()
    test_1d_pca()
    # test_cuda()
    