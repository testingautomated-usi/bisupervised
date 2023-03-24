import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

# NUM_VAL_IMAGES = 1000
NUM_VAL_IMAGES = 50000


def _get_imagenet_val_set():
    dataset = tfds.load('imagenet2012', data_dir="/root/tensorflow_datasets/",
                        download=False, shuffle_files=False,
                        as_supervised=True)["validation"]

    dataset = dataset.take(NUM_VAL_IMAGES)

    # Fetch labels
    labels = []
    for y in tqdm(dataset.map(lambda x, y: y), desc="Fetching ground truth labels"):
        labels.append(y.numpy())

    # Image stream
    images = dataset.map(lambda x, y: x).batch(1).prefetch(100)

    return images, labels


def _make_predictions(model, images, labels, model_name):
    time_start = time.time()
    softmax = model.predict(images)
    pred = np.argmax(softmax, axis=1)
    pred_time = (time.time() - time_start) / pred.shape[0]

    # Compute the time it takes to read the softmax
    time_start = time.time()
    sm_confidences = np.take_along_axis(softmax, pred[:, None], axis=-1)
    supervisor_time = (time.time() - time_start) / pred.shape[0]

    np.testing.assert_array_equal(sm_confidences.flatten(), np.max(softmax, axis=1).flatten())

    res_frame = pd.DataFrame({'index': list(range(pred.shape[0])),
                              'prediction': pred.flatten(),
                              'ground_truth': labels,
                              'sm_confidence': sm_confidences.flatten(),
                              'pred_time': pred_time,
                              'supervisor_time': supervisor_time})

    res_frame.to_csv(f"/generated/predictions_and_uncertainties/imagenet_{model_name}.csv", index=False)


def make_predictions():
    images, labels = _get_imagenet_val_set()

    remote_model = tf.keras.applications.EfficientNetV2L()
    _make_predictions(remote_model, images, labels, "remote")
    del remote_model

    local_model = tf.keras.applications.MobileNetV3Small()
    _make_predictions(local_model, images, labels, "local")
    del local_model


if __name__ == "__main__":
    make_predictions()
