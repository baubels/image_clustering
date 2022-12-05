from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding
from tensorflow.keras.preprocessing import image_dataset_from_directory
from yellowbrick.cluster import KElbowVisualizer
import tensorflow.compat.v2 as tf
import numpy as np



def load_dataset(dir: str = 'caltech-101/101_ObjectCategories/', image_size: tuple[int, int] = (96, 96), batch_size: int = 32) -> tf.data.Dataset:
    """Loads the Caltech-101 Dataset downloaded in directory `dir`.

    Args:
        dir: The directory where the caltech-101 dataset is stored. Defaults to 'caltech-101/101_ObjectCategories/'.
        image_size: The width and height of images to rescale directory dataset to.
        batch_size: The size of batches. Change depending on computational performance.

    Returns:
        A Tensorflow BatchDataset.
    """
    try:
        caltech_dataset = image_dataset_from_directory(
            directory=dir,
            labels='inferred',
            label_mode='categorical',
            batch_size=batch_size,
            image_size=image_size,
        )
    except Exception as ex:
        print("Please ensure you have downloaded the Caltech-101 Dataset from https://data.caltech.edu/records/mzrjq-6wc02,")
        print(f"and placed it into {dir}, or any different directory.")
        raise ex
    return caltech_dataset


def normalise_features(features: np.ndarray) -> np.ndarray:
    """Normalise descriptor features to have mean = 0 and sd = 1.

    Args:
        hogs: A nxp sized NumPy array of HOG descriptors.

    Returns:
        np.ndarray: A nxp sized NumPy array of HOG descriptors with normalised features.
    """
    for c in range(features.shape[1]):
        features[:, c] -= features[:, c].mean()
        if features[:, c].std() != 0:
            features[:, c] /= features[:, c].std()
    return features


def reduce_with_PCA(data: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Do PCA on data, keeping the top `n_components`.

    Args:
        data: A nxp sized NumPy array of data to do PCA with.
                           Features are assumed to be normalised.
                           The Euclidean distance metric is used.
        n_components: The number of most varied features to keep. Defaults to 50.

    Returns:
        A nxn_components sized NumPy array of PCA reduced data.
    """
    pca_reduction = PCA(n_components=n_components)
    return pca_reduction.fit_transform(data)


def reduce_with_Spectral(data: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Do Spectral Embedding on data, keeping the top `n_components`.

    Args:
        data: A nxp sized NumPy array of data to do PCA with.
                           Features are assumed to be normalised.
                           The Euclidean distance metric is used.
        n_components: The number of most varied features to keep. Defaults to 50.

    Returns:
        A nxn_components sized NumPy array of PCA reduced data.
    """
    spectral = SpectralEmbedding(n_components=n_components)
    return spectral.fit_transform(data)


def reduce_with_LLE(data: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Do Locally Linear Embedding on data, keeping the top `n_components`.

    Args:
        data: A nxp sized NumPy array of data to do PCA with.
                           Features are assumed to be normalised.
                           The Euclidean distance metric is used.
        n_components: The number of most varied features to keep. Defaults to 50.

    Returns:
        A nxn_components sized NumPy array of PCA reduced data.
    """
    lle = LocallyLinearEmbedding(n_components=n_components)
    return lle.fit_transform(data)


def KMeans_Elbow(data: np.ndarray, cluster_range: tuple[int, int] = (2, 100)):
    """Apply KMeans to data `cluster_range[1]-cluster_range[0]` times.
       Compute and plot cluster distortions for each cluster count in `cluster_range`.
       Predict optimal cluster count based on the elbow method.

    Args:
        data: A nxp sized NumPy array of data to apply KMeans clustering to.
        cluster_range: The range of clusters of consider for KMeans and analysis. Defaults to (2,100).

    More information on the KElbowVisualizer and arguments available can be found here:
        https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
    """
    print("Applying a KMeans Elbow test by computing cluster distortions.")
    print(f"Input Size: {data.shape}\t Cluster Range to test: {cluster_range}")

    model = KMeans()
    visualizer = KElbowVisualizer(model, k=cluster_range)
    visualizer.fit(data)
    visualizer.show()


def KMeans_clustering(data: np.ndarray, n_clusters: int = 45) -> tuple[KMeans, np.ndarray]:
    """Apply KMeans clustering to `data` using `n_clusters` clusters.

    Args:
        data: A nxp sized NumPy array.
        n_clusters: The number of clusters to use in the KMeans algorithm. Defaults to 45.

    Returns:
        A tuple containing:
            A trained KMeans model on `data` using `n_clusters` clusters.
            A (data.shape[0],) sized NumPy array denoting the cluster number each datapoint is assigned to.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_fit = kmeans.fit(data)
    return kmeans_fit, kmeans.labels_
