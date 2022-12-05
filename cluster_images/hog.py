from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.compat.v2 as tf
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, LocallyLinearEmbedding
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
from cluster_images.hog_configs import *


def load_dataset(dir:str='caltech-101/101_ObjectCategories/', image_size:tuple[int]=(96,96), batch_size:int=32) -> tf.data.Dataset:
    """Loads the Caltech-101 Dataset downloaded in directory `dir`.

    Args:
        dir (str): The directory where the caltech-101 dataset is stored. Defaults to 'caltech-101/101_ObjectCategories/'.
        image_size (int x int): The width and height of images to rescale directory dataset to.
        batch_size (int): The size of batches. Change depending on computational performance.

    Returns:
        tf.data.Dataset: A Tensorflow BatchDataset.
    """
    try:
        caltech_dataset = image_dataset_from_directory(directory=dir,
                                                labels='inferred',
                                                label_mode='categorical',
                                                batch_size=batch_size,
                                                image_size=image_size)
    except Exception as ex: # make this more precise - remove the `bare except`
        print("Please ensure you have downloaded the Caltech-101 Dataset from https://data.caltech.edu/records/mzrjq-6wc02,")
        print(f"and placed it into {dir}, or any different directory.")
        raise ex
    return caltech_dataset


def create_HOG_descriptors(dataset:tf.data.Dataset) -> np.ndarray:
    """Creates HOG descriptors of an input dataset.

    Args:
        dataset: A batched Tensorflow dataset. <- remove typehints from function definitions

    Returns:
        np.ndarray: A nxp sized NumPy array were each row denotes a HOG descriptor of one input image.
    """
    hogs = []
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    # add comments explaining why, so that double for loop != bad
    for x,_ in dataset:  # try squeezing;
        for x_item in x: # nested for loop, try and rid of this!
            item = np.round(x_item.numpy()).astype(np.uint8)
            h = hog.compute(item,winStride,padding,locations)
            hogs.append(h.reshape(1,-1))
    return np.concatenate(hogs, axis=0)


def normalise_features(features:np.ndarray) -> np.ndarray:
    """Normalise descriptor features to have mean = 0 and sd = 1.

    Args:
        hogs (np.ndarray): A nxp sized NumPy array of HOG descriptors.

    Returns:
        np.ndarray: A nxp sized NumPy array of HOG descriptors with normalised features.
    """
    for c in range(features.shape[1]):
        features[:,c] -= features[:,c].mean()
        if features[:,c].std() != 0:
            features[:,c] /= features[:,c].std()
    return features


def reduce_with_PCA(data:np.ndarray, n_components:int=50) -> np.ndarray:
    """Do PCA on data, keeping the top `n_components`.

    Args:
        data (np.ndarray): A nxp sized NumPy array of data to do PCA with.
                           Features are assumed to be normalised.
                           The Euclidean distance metric is used.
        n_components (int, optional): The number of most varied features to keep. Defaults to 50.

    Returns:
        np.ndarray: A nxn_components sized NumPy array of PCA reduced data.
    """
    PCA_reduction = PCA(n_components=n_components)
    return PCA_reduction.fit_transform(data)


def reduce_with_Spectral(data:np.ndarray, n_components:int=50) -> np.ndarray:
    """Do Spectral Embedding on data, keeping the top `n_components`.

    Args:
        data (np.ndarray): A nxp sized NumPy array of data to do PCA with.
                           Features are assumed to be normalised.
                           The Euclidean distance metric is used.
        n_components (int, optional): The number of most varied features to keep. Defaults to 50.

    Returns:
        np.ndarray: A nxn_components sized NumPy array of PCA reduced data.
    """
    Spectral = SpectralEmbedding(n_components=n_components)
    return Spectral.fit_transform(data)


def reduce_with_LLE(data:np.ndarray, n_components:int=50) -> np.ndarray:
    """Do Locally Linear Embedding on data, keeping the top `n_components`.

    Args:
        data (np.ndarray): A nxp sized NumPy array of data to do PCA with.
                           Features are assumed to be normalised.
                           The Euclidean distance metric is used.
        n_components (int, optional): The number of most varied features to keep. Defaults to 50.

    Returns:
        np.ndarray: A nxn_components sized NumPy array of PCA reduced data.
    """
    LLE = LocallyLinearEmbedding(n_components=n_components)
    return LLE.fit_transform(data)


def KMeans_Elbow(data:np.ndarray, cluster_range:tuple[int]=(2,100)):
    """Apply KMeans to data `cluster_range[1]-cluster_range[0]` times. 
       Compute and plot cluster distortions for each cluster count in `cluster_range`.
       Predict optimal cluster count based on the elbow method.

    Args:
        data (np.ndarray): A nxp sized NumPy array of data to apply KMeans clustering to.
        cluster_range (tuple, optional): The range of clusters of consider for KMeans and analysis. Defaults to (2,100).
    
    More information on the KElbowVisualizer and arguments available can be found here:
        https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
    """
    print("Applying a KMeans Elbow test by computing cluster distortions.")
    print(f"Input Size: {data.shape}\t Cluster Range to test: {cluster_range}")

    model = KMeans()
    visualizer = KElbowVisualizer(model, k=cluster_range)
    visualizer.fit(data)
    visualizer.show()


def KMeans_clustering(data:np.ndarray, n_clusters:int=45) -> tuple[KMeans, np.ndarray]:
    """Apply KMeans clustering to `data` using `n_clusters` clusters.

    Args:
        data (np.ndarray): A nxp sized NumPy array.
        n_clusters (int, optional): The number of clusters to use in the KMeans algorithm. Defaults to 45.

    Returns:
        KMeans: A trained KMeans model on `data` using `n_clusters` clusters.
        np.ndarray: A (data.shape[0],) sized NumPy array denoting the cluster number each datapoint is assigned to.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_fit = kmeans.fit(data)
    return kmeans_fit, kmeans.labels_
