"""
This code creates downsampled versions of the original exeter fmri files,
it saves them in a new directory and does not replace them.
This is done prior to calling the main icw_gan brainpedia code,
because of memory issues and contraints with tensor -> array conversion.
This method of downsampling produces nicer images than the
neuroimaging library one.

"""
import math
import os

import scipy
import tensorflow as tf
import numpy as np

# currently using 2D downsampling twice to obtain 3D downsampling, resolution is not great.
from scipy.ndimage import zoom


def downsample(image, x=53, y=63, z=46): # .npy format rather than nifti
    # use resizing twice to  avoid cropping, first on x,y, then on x,z, preserve aspect ratio needs to be false other output size altered
    image = tf.image.resize(image, [x, y], method='nearest', preserve_aspect_ratio=False)
    image = tf.transpose(image, [0,2,1])
    # need to swap y and z dimensions around then perform resize on [53,46]
    image = tf.image.resize(image, [x, z], method='nearest', preserve_aspect_ratio=False)
    image = tf.transpose(image, [0,2,1])
   # print("Image has been downsampled to ", resampled_img.shape)
    # convert from tensor to an array
    print("Type of the resampled image ", type(image)) # this needs to be converted back to an array, is there the same resize operation for numpy arrays
    return image


from itertools import combinations
from functools import reduce
import numpy as np

import SimpleITK as sitk
import cv2
import numpy as np

"""
def resize_and_scale_uint8(image, new_size, outside_pixel_value=0):
    
    Resize the given image to the given size, with isotropic pixel spacing
    and scale the intensities to [0,255].

    Resizing retains the original aspect ratio, with the original image centered
    in the new image. Padding is added outside the original image extent using the
    provided value.

    :param image: A SimpleITK image.
    :param new_size: List of ints specifying the new image size.
    :param outside_pixel_value: Value in [0,255] used for padding.
    :return: a 2D SimpleITK image with desired size and a pixel type of sitkUInt8
    
    # Rescale intensities if scalar image with pixel type that isn't sitkUInt8.
    # We rescale first, so that the zero padding makes sense for all original image
    # ranges. If we resized first, a value of zero in a high dynamic range image may
    # be somewhere in the middle of the intensity range and the outer border has a
    # constant but arbitrary value.
    if image.GetNumberOfComponentsPerPixel() == 1 and image.GetPixelID() != sitk.sitkUInt8:
        final_image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)
    else:
        final_image = image
    new_spacing = [((osz - 1) * ospc) / (nsz - 1) \
                   for ospc, osz, nsz in zip(final_image.GetSpacing(), final_image.GetSize(), new_size)]
    new_spacing = [max(new_spacing)] * final_image.GetDimension()
    center = final_image.TransformContinuousIndexToPhysicalPoint([sz / 2.0 for sz in final_image.GetSize()])
    new_origin = [c - c_index * nspc for c, c_index, nspc in zip(center, [sz / 2.0 for sz in new_size], new_spacing)]
    final_image = sitk.Resample(final_image, size=new_size, outputOrigin=new_origin, outputSpacing=new_spacing,
                                defaultPixelValue=outside_pixel_value)
    return final_image

"""




def resizeMRI(img, shape, mode='nearest', orig_shape=None, order=5):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """

    if orig_shape == None: orig_shape = img.shape

    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0]/orig_shape[0],
        shape[1]/orig_shape[1],
        shape[2]/orig_shape[2]
    )

    # Resize to the given shape
    return zoom(img, factors, mode=mode, order=order)




# takes in the original exeter dataset
filename1 = 'C://Users//hlw69//Documents//temp_fmri_exeter//fmri_npy//'

"""  for fn in os.listdir(filename1):
    print(fn)
    image = np.load(filename1 + fn)
    downsampled = downsample(image)
    downsample_dir = 'C://Users//hlw69//Documents//temp_fmri_exeter//downsample_npy//'
    array_version = downsampled.numpy()
    print(type(array_version), array_version.shape)
        #os.mkdir(downsample_dir)
    np.save(downsample_dir + fn, array_version)
    # converts to tensor to perform downsample
    # converts back to array
    # saves to new folder of downsampled version
    """
import nibabel
from nilearn import plotting
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(30, 10))
print("Axes are ", axes, fig)
for fn in os.listdir(filename1):

    original = np.load(filename1+'44_place_perceive_30.npy')
    original = nibabel.Nifti1Image(original, affine=np.eye(4))
    image = np.load(filename1 + fn)
    first_downsample = downsample(image).numpy()
    first_downsample = nibabel.Nifti1Image(first_downsample, affine=np.eye(4))

    new_downsample = resizeMRI(img = image, shape = [53,63,46],mode='nearest', orig_shape=None, order=3)
    print(new_downsample.shape)
    new_downsample = nibabel.Nifti1Image(new_downsample, affine=np.eye(4))
    print(original.shape, first_downsample.shape, new_downsample.shape)

    matlab = scipy.io.loadmat(filename1+'new_img4.mat')
    print(matlab)
    matlab = nibabel.Nifti1Image(matlab['new_img'], affine=np.eye(4))

    display = plotting.plot_anat(first_downsample,axes=axes[0])
    display = plotting.plot_anat(original,axes=axes[1])
    display = plotting.plot_anat(new_downsample, axes=axes[2])
    display = plotting.plot_anat(matlab, axes=axes[3])
    fig.savefig("my_test_img6.png")
    break
   # plotting.plot_img(fig)
    #plotting.show()