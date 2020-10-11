# %%
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Dict, List, Tuple
from tensorflow import keras
from tensorflow.keras import layers

N_C = 5
N_EPOCHS = 50
INPUT_DIM = 10


emoticons = {
    "0": ":heart:",
    "1": ":baseball:",
    "2": ":smile:",
    "3": ":disappointed:",
    "4": ":fork_and_knife:",
}


def build_model(input_dim: Tuple[int], embed_matrix: np.ndarray) -> keras.Model:
    inputs = layers.Input(shape=input_dim, dtype=tf.int32)
    embedding = layers.Embedding(
        input_dim=embed_matrix.shape[0], output_dim=embed_matrix.shape[1], trainable=False
    )
    embedding.build((None,))
    embedding.set_weights([embed_matrix])
    x = embedding(inputs)
    x = layers.LSTM(units=128, return_sequences=True)(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.LSTM(units=128)(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(units=5)(x)
    outputs = layers.Activation('softmax')(x)

    return keras.Model(inputs, outputs)


def build_dataset(filepath: str, mapping: Dict[str, int]) -> tf.data.Dataset:
    df = pd.read_csv(filepath, header=None)
    xs = [vectorize_sentence(s, mapping, INPUT_DIM) for s in df[0].values]
    ys = df[1].values
    ds = tf.data.Dataset.from_tensor_slices((xs, ys))
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, N_C, dtype=tf.int32)))
    return ds


def vectorize_sentence(sentence: str, mapping: Dict[str, int], width: int) -> List[int]:
    indices = [mapping[word] for word in sentence.lower().split()]
    for _ in range(width - len(indices)):
        indices.append(0)
    return indices


def build_embed_matrix(filepath: str) -> Tuple[Dict[str, int], np.ndarray]:
    df = pd.read_csv(filepath, header=None, delim_whitespace=True, engine='c')
    df = df.sort_values(by=[0])
    vocab = df.pop(0).values
    word_to_index = dict([(w, i) for i, w in enumerate(vocab)])
    embeddings = np.array(df, dtype=np.float64)
    return word_to_index, embeddings


def sample(model, word_to_index, data):
    input = np.array([vectorize_sentence(s, word_to_index, INPUT_DIM) for s in data])
    preds = np.argmax(model.predict(input), axis=1)
    for sent, pred in zip(data, preds):
        print(f'{sent} {emoticons[str(pred)]}')


def main():
    print('Loading word embeddings ...')
    word_to_index, embed_matrix = build_embed_matrix('./data/glove.6B.50d.txt')

    print('Loading dataset ...')
    train_ds = build_dataset('./data/train_emoji.csv', word_to_index)
    train_ds = train_ds.shuffle(1000).batch(32)
    val_ds = build_dataset('./data/tesss.csv', word_to_index).batch(32)

    print('Building model ...')
    model = build_model((INPUT_DIM,), embed_matrix)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    model.summary()

    print('Training model ...')
    _ = model.fit(train_ds, epochs=N_EPOCHS)
    val_perf = model.evaluate(val_ds)
    performance = dict(zip(model.metrics_names, val_perf))
    print(f'Performance: {performance}')

    sample(model, word_to_index, ['I did not have breakfast'])


if __name__ == '__main__':
    main()
