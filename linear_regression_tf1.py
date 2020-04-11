# %%
# lineaer_regression
#
# - deferred execution
# - input placeholders
# - custom model
# - custom training loop

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# hyperparameters
learning_rate = 0.01
n_epochs = 30000


class LinRegModel(object):
    def __init__(self):
        self.w = tf.get_variable('weights', initializer=tf.constant(0.0))
        self.b = tf.get_variable('bias', initializer=tf.constant(0.0))

    def inference(self, x):
        return self.w * x + self.b

    def loss(self, x, y):
        preds = self.inference(x)
        return tf.losses.huber_loss(y, preds)


def main():
    # dataset
    data = np.loadtxt(
        './datasets/birth_life_2010.txt', delimiter='\t', skiprows=1, usecols=(1, 2), unpack=True
    )
    x_train = data[0, :].reshape(1, -1)
    y_train = data[1, :].reshape(1, -1)
    print(f'x: {x_train.shape}, y: {y_train.shape}')

    # inputs
    x = tf.placeholder(tf.float32, shape=[1, None], name='x')
    y = tf.placeholder(tf.float32, shape=[1, None], name='y')

    # graph
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    model = LinRegModel()
    loss = model.loss(x, y)
    train_op = optimizer.minimize(loss)
    losses = []

    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(n_epochs):
            _, l = sess.run([train_op, loss], {x: x_train, y: y_train})
            if (step + 1) % 100 == 0:
                losses.append(l)
        y_pred = sess.run(model.inference(x), {x: x_train})

    # summary
    plt.plot(x_train[0, :], y_train[0, :], 'bo', label='Real data')
    plt.plot(x_train[0, :], y_pred[0, :], 'r', label='Predicted data')
    plt.legend()
    plt.show()

    plt.plot(np.squeeze(losses))
    plt.title('Learning rate')
    plt.xlabel('iterations (per hundreds)')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    main()


# %%
