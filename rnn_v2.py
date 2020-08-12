# %%
import numpy as np
import tensorflow as tf
import io
import time


## layers


class Dense(object):
    def __init__(self, units, input_dim, activation=None, name='dense'):
        self.units = units
        self.activation = activation if activation else lambda x: x
        with tf.name_scope(name):
            self.w = tf.Variable(
                tf.random.uniform([input_dim, units], dtype=tf.float64) * 0.01, name='w'
            )
            self.b = tf.Variable(tf.zeros([1, units], dtype=tf.float64), name='b')
        self.trainable_variables = [self.w, self.b]

    def __call__(self, x):
        z = x @ self.w + self.b
        a = self.activation(z)
        return a


class SimpleRnn(object):
    """Fully connected RNN layer where output is fed back to input
    
    Call Arguments:
    * x - input tensor with shape (m, t_x, n_x]
    * a0 - initial state tensor used by the first cell [m, n_a]
    """

    def __init__(
        self,
        units,
        input_dim,
        activation=tf.nn.tanh,
        return_sequences=False,
        return_state=False,
        name='simplernn',
    ):
        self.units = units
        self.activation = activation if activation else lambda x: x
        self.return_sequences = return_sequences
        self.return_state = return_state
        with tf.name_scope(name):
            self.wax = tf.Variable(
                tf.random.uniform([input_dim, units], dtype=tf.float64) * 0.01, name="wax"
            )
            self.waa = tf.Variable(
                tf.random.uniform([units, units], dtype=tf.float64) * 0.01, name="waa"
            )
            self.ba = tf.Variable(tf.zeros([1, units], dtype=tf.float64), name="ba")
        self.trainable_variables = [self.wax, self.waa, self.ba]

    def __call__(self, x, a0):
        _, t_x, _ = x.shape
        outputs = []
        at = a0
        for t in range(t_x):
            xt = x[:, t, :]
            at = self._step_forward(xt, at)
            outputs.append(at)
        output = tf.stack(outputs, axis=1) if self.return_sequences else outputs[-1]
        return output, at if self.return_state else output

    def _step_forward(self, xt, a_prev):
        zt = a_prev @ self.waa + xt @ self.wax + self.ba
        at = self.activation(zt)
        return at


## model


class RnnModel(object):
    def __init__(self, n_a, vocab_size):
        self.vocab_size = vocab_size
        self.rnn1 = SimpleRnn(n_a, vocab_size, return_sequences=True, return_state=True)
        self.dense1 = Dense(vocab_size, n_a, activation=None)
        self.trainable_variables = self.rnn1.trainable_variables + self.dense1.trainable_variables

    def __call__(self, inputs, a0):
        x, a_last = self.rnn1(inputs, a0)
        outputs = self.dense1(x)
        return outputs, a_last


@tf.function
def train_step(model, optimizer, x, y, a_prev, grad_clip):
    with tf.GradientTape() as tape:
        logits, a_last = model(x, a_prev)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, axis=2)
        loss = tf.reduce_mean(entropy)
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
    x = tf.reshape(x, [1, -1, n_x])
    y = tf.one_hot(y, n_x, dtype=tf.float64)
    y = tf.reshape(y, [1, -1, n_x])
    return x, y


def sample(model, limit, eos, seed=0):
    m = 1
    x = np.zeros([m, 1, model.vocab_size], dtype=np.float64)
    a_prev = np.zeros([m, model.rnn1.units], dtype=np.float64)
    output = []
    for i in range(limit):
        logits, a_next = model(x, a_prev)
        predictions = tf.nn.softmax(logits, axis=2)
        # Sampling is the selection of a value from a group of values,
        # where each value has a probability of being picked.
        # Pick the next character's index according to the probability
        # distribution specified by ŷ⟨t+1⟩
        np.random.seed(i + seed)
        idx = np.random.choice(model.vocab_size, p=predictions[0][-1].numpy().ravel())
        if idx == eos:
            break
        output.append(idx)
        x = np.zeros([m, 1, model.vocab_size], dtype=np.float64)
        x[0, 0, idx] = 1.0
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
    data = io.open('datasets/names2.txt', encoding='utf-8').read().lower()
    vocab = sorted(list(set(data)))
    vocab_size = len(vocab)
    char_to_ix = dict((c, i) for i, c in enumerate(vocab))
    ix_to_char = dict((i, c) for i, c in enumerate(vocab))

    examples = build_dataset(data)

    np.random.seed(0)
    np.random.shuffle(examples)

    # model
    model = RnnModel(N_A, vocab_size)

    # train
    optimizer = tf.optimizers.Adam(LEARNING_RATE)
    writer = tf.summary.create_file_writer(
        'graphs/rnn_tf_v1/lr' + str(optimizer.learning_rate.numpy())
    )
    start_time = time.time()
    total_loss = 0

    a = tf.constant(np.zeros([1, N_A]), dtype=tf.float64)

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
    model = RnnModel(n_a, vocab_size)
    optimizer = tf.optimizers.SGD(0.01)
    x = tf.one_hot([1, 2, 3, 1, 2, 3], vocab_size, dtype=tf.float64)
    y = tf.one_hot([3, 1, 2, 3, 1, 2], vocab_size, dtype=tf.float64)
    x = tf.reshape(x, [1, -1, vocab_size])
    y = tf.reshape(y, [1, -1, vocab_size])
    tf.print('x', x)
    tf.print('y', y[0])
    loss, gradients, a_last = train_step(model, optimizer, x, y, a_prev, 5.0)
    print('a_prev =', a_prev.T)
    print('wax =', tf.transpose(model.rnn1.wax))
    print('waa =', tf.transpose(model.rnn1.waa))
    print(tf.transpose(a_last))
    print("loss =", loss)
    print(f'gradients {gradients[1].shape}')
    print("gradients[\"dWaa\"][1][2] =", gradients[1][2][1])
    output = sample(model, 50, eos=0, seed=0)
    print(f'{output=}')
    output = [str(el) for el in output[:-1]]
    print(''.join(output))


if __name__ == '__main__':
    main()

# %%
