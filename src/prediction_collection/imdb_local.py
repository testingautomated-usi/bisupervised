import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from tqdm import tqdm

MODEL_PATH = "/generated/trained_models/imdb_local"

VOCAB_SIZE = 2000

INPUT_MAXLEN = 100


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


def _train_new_model(ds: str, x_train, y_train) -> tf.keras.Model:
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    num_classes = 2 if ds == "imdb" else 3

    x_train_tf = tf.data.Dataset.from_tensor_slices(x_train)
    vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE,
                                                        output_mode='int',
                                                        output_sequence_length=INPUT_MAXLEN)
    vectorize_layer.adapt(x_train_tf.batch(64))

    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = vectorize_layer(inputs)
    embedding_layer = MyTokenAndPositionEmbedding(INPUT_MAXLEN, VOCAB_SIZE, embed_dim)
    x = embedding_layer(x)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # As opposed to the keras tutorial, we use categorical_crossentropy,
    #   and we run 10 instead of 2 epochs, but with early stopping.
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(25000).batch(32),
        epochs=10,
    )
    return model


def get_model(ds: str, x_train=None, y_train=None) -> tf.keras.Model:
    # Load the model if it exists

    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        assert x_train is not None and y_train is not None
        model = _train_new_model(ds, x_train, y_train)
        model.save(MODEL_PATH)
        return model


if __name__ == '__main__':
    ds = load_dataset("imdb")
    model = get_model("imdb", ds['train']['text'], ds['train']['label'])

    labels = ds['test']['label']
    predictions = []
    sm_confidence = []
    pred_times = []
    superv_times = []

    i = 0
    for x in tqdm(ds['test']['text'], desc="Predicting Local"):
        start = time.time()
        sm_outputs = model.predict([x], verbose=0)[0]
        prediction = np.argmax(sm_outputs)
        end = time.time()

        superv_start_time = time.time()
        confidence = sm_outputs[prediction]
        superv_end_time = time.time()

        predictions.append(prediction)
        sm_confidence.append(confidence)
        pred_times.append(end - start)
        superv_times.append(superv_end_time - superv_start_time)

    res_frame = pd.DataFrame({'id': list(range(len(predictions))),
                              'prediction': predictions,
                              'ground_truth': labels[:len(predictions)],
                              'sm_confidence': sm_confidence,
                              'pred_time': pred_times,
                              'supervisor_time': superv_times})

    res_frame.to_csv(f"/generated/predictions_and_uncertainties/imdb_local.csv", index=False)
