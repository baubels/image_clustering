import cv2
import numpy as np
import tensorflow.compat.v2 as tf

from cluster_images.hog_configs import *



def create_HOG_descriptors(dataset: tf.data.Dataset) -> np.ndarray:
    """Creates HOG descriptors of an input dataset.

    Args:
        dataset: A batched Tensorflow dataset. <- remove typehints from function definitions

    Returns:
        np.ndarray: A nxp sized NumPy array were each row denotes a HOG descriptor of one input image.
    """
    hogs = []
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # this double for loop is necessary
    for x, _ in dataset:
        for x_item in x:
            item = np.round(x_item.numpy()).astype(np.uint8)
            h = hog.compute(item, winStride, padding, locations)
            hogs.append(h.reshape(1, -1))
    return np.concatenate(hogs, axis=0)
