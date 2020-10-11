# %%
import argparse
import contextlib
import io
import os
import time

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

NAME = 'gru_v2'


## model


def build_model(units, timesteps, vocab_size):
    inputs = layers.Input(shape=(timesteps, vocab_size))

    x = layers.GRU(units, return_sequences=True)(inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.GRU(units)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(vocab_size)(x)
    outputs = layers.Activation('softmax')(x)

    return keras.Model(inputs, outputs, name=NAME)


## dataset


def build_dataset(data, vocab, timesteps, stride):
    x = []
    y = []
    for i in range(0, len(data) - timesteps, stride):
        data_slice = data[i : i + timesteps]
        x.append([vocab[c] for c in data_slice])
        y.append(vocab[data[i + timesteps]])
    return tf.data.Dataset.from_tensor_slices((x, y))


## sampling


def sample(model, input, limit, vocab_size, t_x, seed=0):
    output = list(input)
    sentence = [0 for i in range(t_x)]
    sentence[0 : len(input)] = input
    for _ in range(limit):
        x = sentence
        x = tf.one_hot(x, vocab_size, axis=-1)
        x = tf.reshape(x, [1, -1, vocab_size])
        predictions = model.predict(x)
        idx = np.random.choice(vocab_size, p=predictions[0])
        output.append(idx)
        sentence = sentence[1:] + [idx]
        seed += 1
    return output


class Sampler(keras.callbacks.Callback):
    def __init__(self, input, vocab_size, char_to_ix, ix_to_char):
        super(Sampler, self).__init__()
        self.input = [char_to_ix[c] for c in input]
        self.vocab_size = vocab_size
        self.ix_to_char = ix_to_char

    def on_epoch_end(self, epoch, logs=None):
        output = sample(self.model, self.input, 3 * T_X, self.vocab_size, T_X)
        sentence = ''.join([self.ix_to_char[idx] for idx in output])
        print('\n', sentence, '\n')


## utils


def parse_args():
    parser = argparse.ArgumentParser(description=NAME)
    # storage paths
    parser.add_argument(
        '--data_dir', type=str, default='/tmp', help='The location of the input data.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp',
        help='The location of the model checkpoint files.',
    )
    parser.add_argument(
        '--download',
        action='store_true',
        default=False,
        help='Whether to download data to `--data_dir`.',
    )
    # training
    parser.add_argument(
        '--train_epochs',
        type=int,
        default=1,
        help='The number of epochs used to train.',
    )
    parser.add_argument(
        '--epochs_between_evals',
        type=int,
        default=1,
        help='The number of training epochs to run between evaluations.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training and evaluation.',
    )
    # distribution
    parser.add_argument(
        '--distribution_strategy',
        type=str,
        default='mirrored',
        help='The Distribution Strategy to use for training.',
    )
    parser.add_argument(
        '--num_gpus', type=int, default=1, help='How many GPUs to use at each worker .'
    )
    parser.add_argument(
        '--tpu', type=str, default=None, help='The Cloud TPU to use for training.'
    )
    return parser.parse_args()


## main


N_A = 128
T_X = 40
LEARNING_RATE = 0.001
SAMPLE_INPUT = 'two households, both alike in dignity, '


def main():
    # args = parse_args()
    args = {
        'model_dir': 'models',
        'train_epochs': 100,
        'batch_size': 32,
        'distribution_strategy': None,
    }
    keras.backend.clear_session()
    export_path = os.path.join(args['model_dir'], 'saved_model')

    # distribution
    if args['distribution_strategy'] == 'tpu':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
        if args['tpu'] not in ('', 'local'):
            tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        strategy_scope = strategy.scope()
    else:
        strategy_scope = contextlib.nullcontext()

    # dataset
    data = io.open('./data/shakespeare.txt', encoding='utf-8').read().lower()
    vocab = sorted(list(set(data)))
    vocab_size = len(vocab)
    char_to_ix = dict((c, i) for i, c in enumerate(vocab))
    ix_to_char = dict((i, c) for i, c in enumerate(vocab))

    ds = build_dataset(data, char_to_ix, T_X, stride=3)
    ds = ds.map(lambda x, y: (tf.one_hot(x, vocab_size, axis=-1), y))
    ds = ds.shuffle(buffer_size=50000).batch(args['batch_size'])

    # model
    with strategy_scope:
        model = build_model(N_A, T_X, vocab_size)
        model.compile(
            optimizer=keras.optimizers.Adam(LEARNING_RATE),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['sparse_categorical_accuracy'],
        )
        model.summary()

    # training
    callbacks = [
        # keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True),
        # keras.callbacks.TensorBoard(log_dir=args.model_dir),
        Sampler(SAMPLE_INPUT, vocab_size, char_to_ix, ix_to_char),
    ]
    _ = model.fit(
        ds,
        epochs=args['train_epochs'],
        verbose=1,
        callbacks=callbacks,
    )
    model.save(export_path, include_optimizer=False)


if __name__ == '__main__':
    main()

# %%
