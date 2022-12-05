import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub



def create_neural_descriptors(dataset: tf.data.Dataset) -> np.ndarray:
    """Download a pre-trained MobileNetV2 and create neural descriptors of the input `dataset`.

    Args:
        dataset: A TensorFlow BatchedDataset. Ensure that the input size is (batch_size, 96, 96, 3).

    Returns:
        A nx1024 NumPy array, where n is the number of dataset items.
    """

    # load model
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    try:
        model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5", trainable=False)
    except Exception as ex:
        print("The MobileNetV2 model couldn't be loaded. Uhh oh! Check internet? Dead link? It's a MobileNetV2_100_96 feature vector model.")
        raise ex

    # create descriptors
    embedding = []
    for x,_ in dataset:
        assert x.shape[1:] == (96,96,3), "Please ensure that your dataset is of RGB images of size (None, 96,96,3)."
        embedding.append(model(x))
    return np.concatenate(embedding, axis=0)
