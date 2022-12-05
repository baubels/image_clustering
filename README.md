# Image Clustering

I implement a few functions to do Image Clustering of a large set of same-scene data. The main approach is to use HOG or a Neural network to construct high-dimensional vector representations of images. Then, PCA or other available dimensionality-reducing schemes are used on those same vectors to remove unnecessary dimensions and improve performance. Finally, KMeans clustering can be done to cluster the images. A function doing an automatic distortion 'elbow'-test can be used to decide for the optimal number of clusters.

#### Sample Usage
Example usage can be found in `usage_and_visuals.ipynb`. HTML docstrings can be found `docs/build/html/index.html`.

```python
import cluster_images.convert as convert
import cluster_images.hog as hog
import cluster_images.neural as neural

ds = convert.load_dataset() # can also use neural.load_dataset()

# create descriptors using MobileNetV2
neural_descriptors = neural.create_neural_descriptors(ds)                    
neural_descriptors = convert.normalise_features(neural_descriptors)       

# reduce dimensionality
neural_LLE = convert.reduce_with_LLE(neural_descriptors, n_components=50)    # can also try .reduce_with_PCA, .reduce_with_Spectral

# construct kmeans clusters and label
neural_LLE_kmeans, neural_LLE_clustering = convert.KMeans_clustering(neural_LLE)
```