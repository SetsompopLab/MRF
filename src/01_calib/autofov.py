"""
Automatic FOV.

@author: myurt@stanford.edu
@author: ssi@mit.edu
@author: schauman@stanford.edu
"""

import time
import numpy as np
import cv2 as cv


def autofov(img, voxel_size, FOV_target=256, blur_kernel=(3, 3), thresh=0.01):
    '''
    shifts = autofov(img, voxel_size, FOV_target, blur_kernel, thresh)

    Inputs:
      img (Array): Input 3D image.
      voxel_size (Float): Voxel size in mm (identical for all directions).
      FOV_target (Float): Field of view in mm (identical for all directions).
      blur_kernel (Tuple of Floats): Gaussian blurring kernel size.
      thresh (Float): Threshold for binary mask generation.

    Returns:
      Shifts (Array of Floats): Shift values to center image.

    TODO:
      For the next version, look at axial and coronal slices as well to be
      orientation robust and deal with edge-cases like headphones.
    '''
    assert list(img.shape) == [img.shape[0]]*3
    assert len(blur_kernel) == 2

    FOV_in = img.shape[0] * voxel_size
    img_sag = np.max(np.abs(img), axis=0)

    print(">> Binary mask generation... ", end="", flush=True)
    start_time = time.perf_counter()

    z = np.abs(img_sag)
    z = z - np.min(z.ravel())
    z = z/np.max(z.ravel())

    blur = cv.GaussianBlur(z,blur_kernel,0)
    thr = cv.threshold(blur,thresh,1,cv.THRESH_BINARY)[1]
    contours = cv.findContours(np.array(thr, dtype="uint8"),
                               cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    cntsSorted = sorted(contours, key=(lambda x: cv.contourArea(x)))
    (x, y, w, h) = cv.boundingRect(cntsSorted[-1])
    anterior = y + h + 1 # (+1) for margin
    superior = x + w + 1 # (+1) for margin

    si_shift = (FOV_in + FOV_target)/2 - (voxel_size * superior)
    ap_shift = (FOV_in + FOV_target)/2 - (voxel_size * anterior)
    shifts = np.array([0, ap_shift, si_shift])

    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time),
          flush=True)

    print(">> Calculated shifts: "
          "(%0.3f mm (not calculated), %0.3f mm, %0.3f mm)" % tuple(shifts),
          flush=True)
    return shifts
