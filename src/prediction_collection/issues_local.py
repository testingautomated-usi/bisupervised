import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.prediction_collection.issues_utils import prepare_datasets

prepare_datasets()

NUM_CLASSES = 3

MODEL_PATH = "/generated/trained_models/issues_local"


# Taken from https://keras.io/examples/nlp/text_classification_with_transformer/
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    """Single transformer layer."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = (
            embed_dim,
            num_heads,
            ff_dim,
            rate,
        )

    # docstr-coverage: inherited
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    # docstr-coverage: inherited
    def get_config(self):
        config = super().get_config().copy()
        config["embed_dim"] = self.embed_dim
        config["num_heads"] = self.num_heads
        config["ff_dim"] = self.ff_dim
        config["rate"] = self.rate
        config["att"] = self.att.get_config()
        config["ffn"] = self.ffn.get_config()
        config["layernorm1"] = self.layernorm1.get_config()
        config["layernorm2"] = self.layernorm2.get_config()
        config["dropout1"] = self.dropout1.get_config()
        config["dropout2"] = self.dropout2.get_config()
        return config

    # docstr-coverage: inherited
    @classmethod
    def from_config(cls, config):
        instance = cls(
            config["embed_dim"], config["num_heads"], config["ff_dim"], config["rate"]
        )
        instance.att = tf.keras.layers.MultiHeadAttention.from_config(config["att"])
        instance.ffn = tf.keras.Sequential.from_config(config["ffn"])
        instance.layernorm1 = tf.keras.layers.LayerNormalization.from_config(
            config["layernorm1"]
        )
        instance.layernorm2 = tf.keras.layers.LayerNormalization.from_config(
            config["layernorm2"]
        )
        instance.dropout1 = tf.keras.layers.Dropout.from_config(config["dropout1"])
        instance.dropout2 = tf.keras.layers.Dropout.from_config(config["dropout2"])
        return instance


# Taken from https://keras.io/examples/nlp/text_classification_with_transformer/
@tf.keras.utils.register_keras_serializable()
class MyTokenAndPositionEmbedding(tf.keras.layers.Layer):
    """Construct the embedding matrix."""

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        # super(MyTokenAndPositionEmbedding, self).__init__()
        super(MyTokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen, self.vocab_size, self.embed_dim = maxlen, vocab_size, embed_dim

    # docstr-coverage: inherited
    def build(self, input_shape):
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.embed_dim
        )
        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=self.maxlen, output_dim=self.embed_dim
        )

    # docstr-coverage: inherited
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    # docstr-coverage: inherited
    def get_config(self):
        return {
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        }


def train_new_model(x_train, y_train, vocab_size, input_maxlen, batch_size=32) -> tf.keras.Model:
    embed_dim = 64  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    # determine class weights
    class_weights = {}
    for i in range(NUM_CLASSES):
        class_weights[i] = len(y_train) / (np.sum(np.array(y_train) == i))

    x_train_tf = tf.data.Dataset.from_tensor_slices(x_train)
    vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size,
                                                        output_mode='int',
                                                        output_sequence_length=input_maxlen)
    vectorize_layer.adapt(x_train_tf.batch(64))

    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = vectorize_layer(inputs)
    embedding_layer = MyTokenAndPositionEmbedding(input_maxlen, vocab_size, embed_dim)
    x = embedding_layer(x)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(100, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(200).batch(batch_size).prefetch(20),
        class_weight=class_weights,
        epochs=10,
    )
    return model


def _load_or_train_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except:
        train_set = pd.read_csv("/generated/datasets/issues-training/github-labels-top3-803k-train.csv")

        _preprocess(train_set)

        model: tf.keras.Model = train_new_model(train_set['x'].tolist(), train_set['labels'].tolist(),
                                                vocab_size=2000, input_maxlen=200, batch_size=320)

        model.save(MODEL_PATH)
        return model


def _preprocess(dataframe):
    dataframe['issue_label'] = pd.Categorical(dataframe['issue_label'])
    dataframe['labels'] = dataframe['issue_label'].cat.codes
    dataframe["issue_title"] = dataframe["issue_title"].fillna("")
    dataframe["issue_body"] = dataframe["issue_body"].fillna("")
    dataframe['x'] = dataframe['issue_title'] + [". "] + dataframe['issue_body']


def collect_predictions():
    model = _load_or_train_model()

    test_set = pd.read_csv("/generated/datasets/issues-test/github-labels-top3-803k-test.csv")

    _preprocess(dataframe=test_set)

    predictions = []
    ground_truth = []
    sm_confidences = []
    pred_times = []
    supervisor_times = []
    for x, y in tqdm(zip(test_set['x'].tolist(), test_set['labels'].tolist()),
                     total=len(test_set['x'].tolist()),
                     desc="Predicting issue labels on local model"):
        time_start = time.time()
        softmax = model.predict([x], verbose=0)[0]
        pred = np.argmax(softmax)
        pred_time = time.time() - time_start

        # Compute the time it takes to read the softmax
        time_start = time.time()
        sm_conf = softmax[pred]
        supervisor_time = time.time() - time_start

        predictions.append(pred)
        ground_truth.append(y)
        sm_confidences.append(sm_conf)
        pred_times.append(pred_time)
        supervisor_times.append(supervisor_time)

    res = pd.DataFrame({'index': list(range(len(predictions))),
                        'prediction': predictions,
                        'ground_truth': ground_truth,
                        'sm_confidence': sm_confidences,
                        'pred_time': pred_times,
                        'supervisor_time': supervisor_times})

    res.to_csv("/generated/predictions_and_uncertainties/issues_local.csv", index=False)


if __name__ == '__main__':
    collect_predictions()
