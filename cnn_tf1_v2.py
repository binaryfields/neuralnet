# %%
# cnn_tf1_v2
#
# - deferred execution
# - tf layers
# - tf estimator training loop

# TODO switch to softmax

import h5py
import numpy as np
import tensorflow as tf2

tf = tf2.compat.v1
tf.disable_v2_behavior()


# hyperparams
learning_rate = 0.001
n_epochs = 500


def model_net(x, n_classes):
    with tf.variable_scope('convnet'):
        x = tf.layers.conv2d(
            x,
            8,
            4,
            padding='same',
            activation=tf.nn.relu,
            use_bias=False,
            kernel_initializer=tf2.keras.initializers.GlorotNormal(seed=0),
        )
        x = tf.layers.max_pooling2d(x, 8, 8)
        x = tf.layers.conv2d(
            x,
            16,
            2,
            padding='same',
            activation=tf.nn.relu,
            use_bias=False,
            kernel_initializer=tf2.keras.initializers.GlorotNormal(seed=0),
        )
        x = tf.layers.max_pooling2d(x, 4, 4)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(
            x, n_classes, kernel_initializer=tf2.keras.initializers.GlorotNormal(seed=0)
        )
        return x


def model_fn(features, labels, mode):
    logits = model_net(features, labels.shape[1])
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, tf.float32)
        )
    )
    predictions = tf.greater(tf.nn.sigmoid(logits), tf.constant(0.5))
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    elif mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
        eval_metric_ops = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops
        )
    else:
        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def load_dataset():
    for (file_name, prefix) in [
        ('images_train.h5', 'train_set'),
        ('images_test.h5', 'test_set'),
    ]:
        model = h5py.File(f'data/{file_name}', 'r')
        x = np.array(model[prefix + '_x'][:], dtype=np.float32)
        x = x / 255
        y = np.array(model[prefix + '_y'][:], dtype=np.int32)
        y = y.reshape((y.shape[0], 1))
        yield (x, y)


def main():
    # ops.reset_default_graph()
    tf.set_random_seed(1)
    # dataset
    (train_ds, test_ds) = load_dataset()
    print('Train X{} Y{}'.format(train_ds[0].shape, train_ds[1].shape))
    print('Test  X{} Y{}'.format(test_ds[0].shape, test_ds[1].shape))
    # train
    input_fn = tf.estimator.inputs.numpy_input_fn(
        train_ds[0], train_ds[1], batch_size=128, shuffle=False, num_epochs=None
    )
    model = tf.estimator.Estimator(model_fn)
    model.train(input_fn, steps=n_epochs)
    # evaluate
    input_fn = tf.estimator.inputs.numpy_input_fn(
        test_ds[0], test_ds[1], batch_size=256, shuffle=False
    )
    metrics = model.evaluate(input_fn)
    print('Test accuracy: {accuracy:0.3f}'.format(**metrics))


if __name__ == '__main__':
    main()


# %%
