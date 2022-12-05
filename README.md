# Image Clustering

I implement a few functions to do Image Clustering of a large set of same-scene data. The main approach is to use HOG or a Neural network to construct high-dimensional vector representations of images. Then, PCA or other available dimensionality-reducing schemes are used on those same vectors to remove unnecessary dimensions and improve performance. Finally, KMeans clustering can be done to cluster the images. A function doing an automatic distortion 'elbow'-test can be used to decide for the optimal number of clusters.

#### Usage

```python
import hog
import neural

ds = hog.load_dataset() # can also use neural.load_dataset()
```

#### create descriptors using MobileNetV2

```python
neural_descriptors = neural.create_neural_descriptors(ds)           # can also use hog.x
neural_descriptors = neural.normalise_features(neural_descriptors)  # can also use hog.x
```

#### reduce dimensionality

```python
neural_LLE = neural.reduce_with_LLE(neural_descriptors, n_components=50) # can also try .reduce_with_PCA, .reduce_with_Spectral
```

### construct kmeans clusters and label

```python
neural_LLE_kmeans, neural_LLE_clustering = neural.KMeans_clustering(neural_LLE)
```

