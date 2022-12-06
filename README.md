# Image Clustering

I implement a few functions to do Image Clustering of a large set of same-scene data. The main approach is to use HOG or a Neural network to construct high-dimensional vector representations of images. Then, PCA or other available dimensionality-reducing schemes are used on those same vectors to remove unnecessary dimensions and improve performance. Finally, KMeans clustering can be done to cluster the images. A function doing an automatic distortion 'elbow'-test can be used to decide for the optimal number of clusters.

Example usage can be found in `usage_and_visuals.ipynb`. HTML docstrings can be found `docs/build/html/index.html`. I am running this successfully with `Python 3.10.8`.

#### Note:
To run the code in the notebooks, the `caltech-101` dataset needs to be downloaded and unzipped in the same folder.
It should have file path structure `caltech-101/101_ObjectCategories/...`. For purposes of speed, I kept the first (alphabetical) 45 image categories (accordian -> gramophone). Once this is done, the code will run. Other datasets may be used, they have to be RGB and images downscalable to (96 x 96 x 3).

#### Sample Usage

```
python3 -m venv cluster_images_venv
source cluster_images_venv/bin/activate
git clone https://github.com/baubels/image_clustering.git
cd image_clustering
pip install -r requirements.txt
```

```python
import cluster_images.convert as convert
import cluster_images.hog as hog
import cluster_images.neural as neural

ds = convert.load_dataset()

# create descriptors using MobileNetV2 (HOG can be used too; hog.create_HOG_descriptors(data))
neural_descriptors = neural.create_neural_descriptors(ds) 
neural_descriptors = convert.normalise_features(neural_descriptors)       

# reduce dimensionality
neural_LLE = convert.reduce_with_LLE(neural_descriptors, n_components=50)    # can also try .reduce_with_PCA, .reduce_with_Spectral

# construct kmeans clusters and label (45 clusters assumed as standard)
neural_LLE_kmeans, neural_LLE_clustering = convert.KMeans_clustering(neural_LLE, n_clusters=45)
```

`neural_LLE_kmeans` is a fitted `KMeans` model, able to predict clusters based on input points.

`neural_LLE_clustering` are the cluster labels for the `neural_LLE` data provided.