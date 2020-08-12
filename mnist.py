# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a simple model on the MNIST dataset."""

import argparse
import contextlib
import os

import tensorflow as tf
import tensorflow_datasets as tfds

NAME = 'minst'


def build_model():
    """Constructs the ML model used to predict handwritten digits."""

    image = tf.keras.layers.Input(shape=(28, 28, 1))

    y = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(image)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(y)
    y = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(1024, activation='relu')(y)
    y = tf.keras.layers.Dropout(0.4)(y)

    probs = tf.keras.layers.Dense(10, activation='softmax')(y)

    return tf.keras.models.Model(image, probs, name=NAME)


@tfds.decode.make_decoder(output_dtype=tf.float32)
def decode_image(example, feature):
    """Convert image to float32 and normalize from [0, 255] to [0.0, 1.0]."""
    return tf.cast(feature.decode_example(example), dtype=tf.float32) / 255


def parse_args():
    parser = argparse.ArgumentParser(description=NAME)
    # storage paths
    parser.add_argument(
        '--data_dir', type=str, default='/tmp', help='The location of the input data.'
    )
    parser.add_argument(
        '--model_dir', type=str, default='/tmp', help='The location of the model checkpoint files.'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        default=False,
        help='Whether to download data to `--data_dir`.',
    )
    # training
    parser.add_argument(
        '--train_epochs', type=int, default=1, help='The number of epochs used to train.'
    )
    parser.add_argument(
        '--epochs_between_evals',
        type=int,
        default=1,
        help='The number of training epochs to run between evaluations.',
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024, help='Batch size for training and evaluation.'
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
    parser.add_argument('--tpu', type=str, default=None, help='The Cloud TPU to use for training.')
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_full_path = os.path.join(args.model_dir, 'model.ckpt-{epoch:04d}')
    export_path = os.path.join(args.model_dir, 'saved_model')

    # distribution strategy
    if args.distribution_strategy == 'tpu':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
        if args.tpu not in ('', 'local'):
            tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        strategy_scope = strategy.scope()
    else:
        strategy_scope = contextlib.nullcontext()

    # dataset
    mnist = tfds.builder('mnist', data_dir=args.data_dir)
    if args.download:
        mnist.download_and_prepare()

    num_train_examples = mnist.info.splits['train'].num_examples
    num_eval_examples = mnist.info.splits['test'].num_examples

    mnist_train, mnist_test = mnist.as_dataset(
        split=['train', 'test'],
        decoders={'image': decode_image()},  # pylint: disable=no-value-for-parameter
        as_supervised=True,
    )
    mnist_train = mnist_train.cache().repeat().shuffle(buffer_size=50000).batch(args.batch_size)
    mnist_test = mnist_test.cache().repeat().batch(args.batch_size)

    # model
    with strategy_scope:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.05, decay_steps=100000, decay_rate=0.96
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        model = build_model()
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'],
        )
        model.summary()

    # training
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=args.model_dir),
    ]

    history = model.fit(
        mnist_train,
        epochs=args.train_epochs,
        steps_per_epoch=num_train_examples // args.batch_size,
        callbacks=callbacks,
        validation_steps=num_eval_examples // args.batch_size,
        validation_data=mnist_test,
        validation_freq=args.epochs_between_evals,
    )

    model.save(export_path, include_optimizer=False)

    eval_output = model.evaluate(mnist_test, steps=num_eval_examples // args.batch_size, verbose=2)

    # stats = common.build_stats(history, eval_output, callbacks)


if __name__ == '__main__':
    main()
