# %%
# rnn_tf1_v1
#
# Description:
#
# - At each time-step, the RNN tries to predict what is the next character given the previous characters
# - ŷ⟨t+1⟩[i] represents the probability that the character indexed by "i" is the next character.
#
# RNN Cell:
#
# a⟨t⟩ = tanh(Waa * a⟨t−1⟩ + Wax * x⟨t⟩ + ba)
# ŷ⟨t⟩ = softmax(Wya * a⟨t⟩ + by)
#
# Definitions:
#
# n_x - number of units in a single time step of a single training example
# n_y - number of units in the vector represeting predictions
# t_x - number of time steps
# m - number of training examples in a batch
# input - input mini-batch size of m training examples (m, n_x, t_x)
# xt - the current time step input data, 2d slice for time step t (m, n_x)
# a_prev - hidden state passed from one time step to another (m, n_a)
# wax - weights relating input to activation/hidden state (n_x, n_a)
# waa - weights relating previous activation to hidden state (n_a, n_a)
# wya - weights relating activation/hidden state to output (n_a, n_y)
# yt - predictions for time step t (m, n_y)
# output - (m, n_y, t_x)
#

import time
import tensorflow as tf
import numpy as np
import io
import os


## model


class RnnModel(object):
    def __init__(self, n_a, n_x, n_y, test=False):
        self.n_a = n_a
        self.n_x = n_x
        if not (test):
            self.wax = tf.Variable(
                tf.random.uniform([n_x, n_a], dtype=tf.float64) * 0.01, name="wax"
            )
            self.waa = tf.Variable(
                tf.random.uniform([n_a, n_a], dtype=tf.float64) * 0.01, name="waa"
            )
            self.wya = tf.Variable(
                tf.random.uniform([n_a, n_y], dtype=tf.float64) * 0.01, name="wya"
            )
            self.ba = tf.Variable(tf.zeros([1, n_a], dtype=tf.float64), name="ba")
            self.by = tf.Variable(tf.zeros([1, n_y], dtype=tf.float64), name="by")
        else:
            self.wax = tf.Variable(
                np.random.randn(n_a, n_x).T, dtype=tf.float64, name="wax"
            )
            self.waa = tf.Variable(
                np.random.randn(n_a, n_a).T, dtype=tf.float64, name="waa"
            )
            self.wya = tf.Variable(
                np.random.randn(n_y, n_a).T, dtype=tf.float64, name="wya"
            )
            self.ba = tf.Variable(np.random.randn(n_a, 1).T, dtype=tf.float64, name="ba")
            self.by = tf.Variable(np.random.randn(n_y, 1).T, dtype=tf.float64, name="by")
        self.trainable_variables = [self.wax, self.waa, self.wya, self.ba, self.by]

    def inference(self, x, a0):
        _, _, t_x = x.shape
        output = []
        at = a0
        for t in range(t_x):
            xt = x[:, :, t]
            zt, at = self._step_forward(xt, at)
            output.append(zt)
        output = tf.stack(output, axis=2)
        return output, at

    def predict(self, xt, a_prev):
        zt, a_next = self._step_forward(xt, a_prev)
        yt = tf.nn.softmax(zt)
        return yt, a_next

    def _step_forward(self, xt, a_prev):
        at = tf.tanh(tf.matmul(xt, self.wax) + tf.matmul(a_prev, self.waa) + self.ba)
        zt = tf.matmul(at, self.wya) + self.by
        return zt, at


@tf.function
def train_step(model, optimizer, x, y, a_prev, grad_clip):
    with tf.GradientTape() as tape:
        logits, a_last = model.inference(x, a_prev)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, axis=1)
        loss = tf.reduce_sum(entropy)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, gradients, a_last


## dataset


def build_dataset(data):
    return [x.lower().strip() for x in data.split('\n')]


def vectorize(example, n_x, char_to_ix):
    x = [char_to_ix[c] for c in example]
    y = x + [char_to_ix['\n']]
    x0 = tf.zeros([1, n_x], dtype=tf.float64)
    x = tf.one_hot(x, n_x, dtype=tf.float64)
    x = tf.concat([x0, x], axis=0)
    x = tf.reshape(tf.transpose(x), [1, n_x, -1])
    y = tf.one_hot(y, n_x, dtype=tf.float64)
    y = tf.reshape(tf.transpose(y), [1, n_x, -1])
    return x, y


def sample(model, limit, eos, seed=0):
    xt = np.zeros([1, model.n_x], dtype=np.float64)
    a_prev = np.zeros([1, model.n_a], dtype=np.float64)
    output = []
    for i in range(limit):
        yt, a_next = model.predict(xt, a_prev)
        # Sampling is the selection of a value from a group of values,
        # where each value has a probability of being picked.
        # Pick the next character's index according to the probability
        # distribution specified by ŷ⟨t+1⟩
        np.random.seed(i + seed)
        idx = np.random.choice(model.n_x, p=yt.numpy().ravel())
        if idx == eos:
            break
        output.append(idx)
        xt = np.zeros([1, model.n_x], dtype=np.float64)
        xt[0, idx] = 1.0
        a_prev = a_next
        seed += 1
    output.append(eos)
    return output


## main


LEARNING_RATE = 0.001
N_ITERS = 50000
N_A = 50
GRAD_CLIP = 5.0
SKIP_STEP = 2000
SKIP_SAMPLES = 7


def main():
    # dataset
    data = io.open('data/names2.txt', encoding='utf-8').read().lower()
    vocab = sorted(list(set(data)))
    vocab_size = len(vocab)
    char_to_ix = dict((c, i) for i, c in enumerate(vocab))
    ix_to_char = dict((i, c) for i, c in enumerate(vocab))

    examples = build_dataset(data)

    np.random.seed(0)
    np.random.shuffle(examples)

    # model
    model = RnnModel(N_A, vocab_size, vocab_size)

    # train
    optimizer = tf.optimizers.Adam(LEARNING_RATE)
    writer = tf.summary.create_file_writer(
        'graphs/rnn_tf_v1/lr' + str(optimizer.learning_rate.numpy())
    )
    start_time = time.time()
    total_loss = 0

    a = tf.constant(np.zeros([1, model.n_a]), dtype=tf.float64)

    for step in range(N_ITERS):
        x, y = vectorize(examples[step % len(examples)], vocab_size, char_to_ix)
        loss, _, a = train_step(model, optimizer, x, y, a, GRAD_CLIP)
        with writer.as_default():
            tf.summary.scalar('loss', loss, step=step)
            writer.flush()
        total_loss += loss
        if step % SKIP_STEP == 0:
            avg_loss = total_loss / SKIP_STEP
            print(f'loss at step {step}: {avg_loss}')
            total_loss = 0
            for i in range(SKIP_SAMPLES):
                output = sample(model, 50, eos=char_to_ix['\n'], seed=i)
                output = output[:-1]
                output = [ix_to_char[idx] for idx in output]
                print(''.join(output))
            print('\n')

    end_time = time.time()
    print(f'total time: {end_time - start_time}s')
    writer.close()


def test():
    np.random.seed(1)
    vocab_size, n_a = 4, 5
    a_prev = np.random.randn(n_a, 1).T
    model = RnnModel(n_a, vocab_size, vocab_size, True)
    optimizer = tf.optimizers.SGD(0.01)
    x = tf.one_hot([1, 2, 3, 1, 2, 3], vocab_size, dtype=tf.float64)
    y = tf.one_hot([3, 1, 2, 3, 1, 2], vocab_size, dtype=tf.float64)
    x = tf.reshape(tf.transpose(x), [1, vocab_size, -1])
    y = tf.reshape(tf.transpose(y), [1, vocab_size, -1])
    tf.print('x', tf.transpose(x))
    tf.print('y', y[0])
    loss, gradients, a_last = train_step(model, optimizer, x, y, a_prev, GRAD_CLIP)
    print('a_prev =', a_prev.T)
    print('wax =', tf.transpose(model.wax))
    print('waa =', tf.transpose(model.waa))
    print(tf.transpose(a_last))
    print("loss =", loss)
    print(f'gradients {gradients[1].shape}')
    print("gradients[\"dWaa\"][1][2] =", gradients[1][2][1])
    output = sample(model, 50, eos=0, seed=0)
    output = [str(el) for el in output[:-1]]
    print(''.join(output))


if __name__ == '__main__':
    main()


# %%
