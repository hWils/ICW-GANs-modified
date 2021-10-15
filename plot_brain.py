from nilearn import plotting, surface
import numpy as np
import nibabel

import nilearn


# for tensors, ,transpose is not usable, there is tensor function for this but only 2D tensors. Permute is the solution here!


import nibabel as nib

from nilearn.image import resample_img
import tensorflow as tf
from nilearn import plotting
synthetic_path = "C://Users//hlw69//Documents//fMRI_transferlearning//Zhang et al//ICW-GANs-master//ICW-GANs-master//synthetic//test_21_0_0.nii.gz" # synthetic
real_path = 'C://Users//hlw69//Documents//temp_fmri_exeter//fmri_npy//33_face_perceive_1.npy'
downsampled_real_path = "C://Users//hlw69//Documents//fMRI_transferlearning//Zhang et al//ICW-GANs-master//ICW-GANs-master//tflib//test_data//data.npy"
real_labels = "C://Users//hlw69//Documents//fMRI_transferlearning//Zhang et al//ICW-GANs-master//ICW-GANs-master//tflib//test_data//labels.npy"
image = np.load(downsampled_real_path)
original_image = np.load(real_path)
#original_image = nibabel.Nifti1Image(original_image, affine=np.eye(4))

# use resizing twice to  avoid cropping, first on x,y, then on x,z, preserve aspect ratio needs to be false other output size altered
resampled_img = tf.image.resize(original_image, [53, 63], method='nearest', preserve_aspect_ratio=False)
swapped = tf.transpose(resampled_img, [0, 2, 1])
# need to swap y and z dimensions around then perform resize on [53,46]
swapped = tf.image.resize(swapped, [53, 46], method='nearest', preserve_aspect_ratio=False)
print(swapped.shape)
resampled_img = tf.transpose(swapped, [0, 2, 1])
print(resampled_img.shape)


def downsample(image): # .npy format rather than nifti
    # use resizing twice to  avoid cropping, first on x,y, then on x,z, preserve aspect ratio needs to be false other output size altered
    image = tf.image.resize(image, [53, 63], method='nearest', preserve_aspect_ratio=False)
    transposed = tf.transpose(image, [0,2,1])
    # need to swap y and z dimensions around then perform resize on [53,46]
    downsampled = tf.image.resize(transposed, [53, 46], method='nearest', preserve_aspect_ratio=False)
    print(swapped.shape)
    resampled_img = tf.transpose(downsampled, [0,2,1])
    print(resampled_img.shape)
    return resampled_img


# then swap back again? Or is it transpose?



    # resample_img(original_image, target_affine=np.eye(4), target_shape=(53, 63, 46), interpolation='linear',
                           #  clip='false')
resampled_img = nibabel.Nifti1Image(resampled_img, affine=np.eye(4))
#print(np.load(real_labels))
nifti_file = nibabel.Nifti1Image(image[2], affine=np.eye(4))
img = nib.load(synthetic_path)
print(img.shape, nifti_file.shape, resampled_img.shape)
#plotting.plot_glass_brain(nifti_file, threshold=3) # does not work as not in mni space

plotting.plot_img(resampled_img)
plotting.show()
"""
fsaverage = nilearn.datasets.fetch_surf_fsaverage()



filename1 = 'C://Users//hlw69//Documents//temp_fmri_exeter//fmri_npy//30_face_perceive_1.npy'
image = np.load(filename1)
nifti_file = nibabel.Nifti1Image(image, affine=np.eye(4))
print(nifti_file.shape)


from nilearn.plotting import plot_epi, show
plot_epi(nifti_file)



texture = surface.vol_to_surf(nifti_file, fsaverage.pial_right)

# Visualizing t-map image on EPI template with manual
# positioning of coordinates using cut_coords given as a list
#plotting.plot_glass_brain(nifti_file,
                       # title="plot_stat_map")

plotting.show()

"""