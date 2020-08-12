# %%
import numpy as np
import tensorflow as tf
import io
import time


## layers


class Layer(object):
    def __init__(self):
        self.trainable_variables = []

    def summary(self):
        n_param = np.sum([np.prod(v.shape.as_list()) for v in self.trainable_variables])
        return n_param


class Activation(Layer):
    def __init__(self, activation=None):
        super(Activation, self).__init__()
        self.activation = activation if activation else lambda x: x

    def __call__(self, x):
        return self.activation(x)


class Dense(Layer):
    def __init__(self, units, input_dim, activation=None, name='dense'):
        super(Dense, self).__init__()
        self.units = units
        self.activation = activation if activation else lambda x: x
        with tf.name_scope(name):
            self.w = tf.Variable(tf.random.uniform([input_dim, units]) * 0.01, name='w')
            self.b = tf.Variable(tf.zeros([1, units]), name='b')
        self.trainable_variables = [self.w, self.b]

    def __call__(self, x):
        z = x @ self.w + self.b
        a = self.activation(z)
        return a


class Dropout(Layer):
    def __init__(self, rate, seed=None):
        super(Dropout, self).__init__()
        self.rate = rate
        self.seed = seed

    def __call__(self, x):
        d = tf.random.uniform(x.shape, seed=self.seed) < (1.0 - self.rate)
        a = x * tf.cast(d, tf.float32)
        a = a / (1.0 - self.rate)
        return a


class Gru(Layer):
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
        recurrent_activation=tf.nn.sigmoid,
        return_sequences=False,
        return_state=False,
        name='gru',
    ):
        super(Gru, self).__init__()
        self.units = units
        self.activation = activation if activation else lambda x: x
        self.recurrent_activation = (
            recurrent_activation if recurrent_activation else lambda x: x
        )
        self.return_sequences = return_sequences
        self.return_state = return_state
        with tf.name_scope(name):
            # cell weights
            self.wcx = tf.Variable(
                tf.random.uniform([input_dim, units]) * 0.01, name="wcx"
            )
            self.wcc = tf.Variable(tf.random.uniform([units, units]) * 0.01, name="wcc")
            self.bc = tf.Variable(tf.zeros([1, units]), name="bc")
            # update gate weights
            self.wux = tf.Variable(
                tf.random.uniform([input_dim, units]) * 0.01, name="wux"
            )
            self.wuc = tf.Variable(tf.random.uniform([units, units]) * 0.01, name="wuc")
            self.bu = tf.Variable(tf.zeros([1, units]), name="bu")
            # relevance gate weights
            self.wrx = tf.Variable(
                tf.random.uniform([input_dim, units]) * 0.01, name="wrx"
            )
            self.wrc = tf.Variable(tf.random.uniform([units, units]) * 0.01, name="wrc")
            self.br = tf.Variable(tf.zeros([1, units]), name="br")
        self.trainable_variables = [
            self.wcx,
            self.wcc,
            self.bc,
            self.wux,
            self.wuc,
            self.bu,
            self.wrx,
            self.wrc,
            self.br,
        ]

    def __call__(self, x, a0):
        m, t_x, _ = x.shape
        outputs = []
        ct = a0 if a0 != None else tf.zeros([m, self.units])
        for t in range(t_x):
            xt = x[:, t, :]
            ct = self._step_forward(xt, ct)
            outputs.append(ct)
        output = tf.stack(outputs, axis=1) if self.return_sequences else outputs[-1]
        if self.return_state:
            return output, ct
        else:
            return output

    def _step_forward(self, xt, c_prev):
        g_u = self.recurrent_activation(c_prev @ self.wuc + xt @ self.wux + self.bu)
        g_r = self.recurrent_activation(c_prev @ self.wrc + xt @ self.wrx + self.br)
        ct_candidate = self.activation(
            (g_r * c_prev) @ self.wcc + xt @ self.wcx + self.bc
        )
        ct = g_u * ct_candidate + (1 - g_u) * c_prev
        return ct


## model


class RnnModel(object):
    def __init__(self, n_a, vocab_size):
        self.vocab_size = vocab_size
        self.rnn1 = Gru(n_a, vocab_size, return_sequences=True, return_state=True)
        self.dropout1 = Dropout(0.25)
        self.rnn2 = Gru(n_a, n_a, return_sequences=False, return_state=False)
        self.dropout2 = Dropout(0.25)
        self.dense1 = Dense(vocab_size, n_a, activation=None)
        self.trainable_variables = (
            [] + self.rnn1.trainable_variables + self.dense1.trainable_variables
        )
        self.layers = [self.rnn1, self.dropout1, self.rnn2, self.dropout2, self.dense1]

    def __call__(self, inputs, a0):
        x, a_last = self.rnn1(inputs, a0)
        x = self.dropout1(x)
        x = self.rnn2(x, None)
        x = self.dropout2(x)
        outputs = self.dense1(x)
        return outputs, a_last

    def summary(self):
        print('Layers')
        print(''.join(['=' for _ in range(40)]))
        total_params = 0
        for layer in self.layers:
            n_param = layer.summary()
            print(f'{type(layer).__name__} {n_param}')
            total_params += n_param
        print(''.join(['=' for _ in range(40)]))
        print(f'Trainable params: {total_params}')


@tf.function
def train_step(model, optimizer, x, y, a_prev, grad_clip):
    with tf.GradientTape() as tape:
        logits, a_last = model(x, a_prev)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(entropy)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, gradients, a_last


## dataset


def build_dataset(data, vocab, t_x, stride):
    for i in range(0, len(data) - t_x, stride):
        x = [vocab[c] for c in data[i : i + t_x]]
        y = vocab[data[i + t_x]]
        yield (x, y)


def sample(model, input, limit, t_x, seed=0):
    output = list(input)
    sentence = [0 for i in range(t_x)]
    sentence[0 : len(input)] = input
    a_prev = None
    for i in range(limit):
        x = sentence
        x = tf.one_hot(x, model.vocab_size, axis=-1)
        x = tf.reshape(x, [1, -1, model.vocab_size])
        logits, a_next = model(x, a_prev)
        predictions = tf.nn.softmax(logits)
        np.random.seed(i + seed)
        idx = np.random.choice(model.vocab_size, p=predictions[0].numpy().ravel())
        output.append(idx)
        sentence = sentence[1:] + [idx]
        a_prev = a_next
        seed += 1
    return output


## main


N_A = 128
T_X = 40

LEARNING_RATE = 0.003
N_EPOCHS = 100  # 1000
BATCH_SIZE = 128
GRAD_CLIP = 5.0
SKIP_STEP = 100
SAMPLE_STEP = 1000

INPUT = 'two households, both alike in dignity, '


def main():
    # dataset
    data = io.open('./datasets/shakespeare.txt', encoding='utf-8').read().lower()
    vocab = sorted(list(set(data)))
    vocab_size = len(vocab)
    char_to_ix = dict((c, i) for i, c in enumerate(vocab))
    ix_to_char = dict((i, c) for i, c in enumerate(vocab))

    def gen_dataset():
        yield from build_dataset(data, char_to_ix, T_X, stride=3)

    dataset = tf.data.Dataset.from_generator(
        generator=gen_dataset,
        output_types=(tf.int32, tf.int32),
        output_shapes=((T_X,), ()),
    )
    dataset = dataset.map(
        lambda x, y: (tf.one_hot(x, vocab_size, axis=-1), tf.one_hot(y, vocab_size))
    )
    dataset = dataset.batch(BATCH_SIZE)

    # model
    model = RnnModel(N_A, vocab_size)
    model.summary()

    # train
    optimizer = tf.optimizers.Adam(LEARNING_RATE)
    writer = tf.summary.create_file_writer(
        'graphs/lstm_v1/lr' + str(optimizer.learning_rate.numpy())
    )
    start_time = time.time()
    step = 0
    a = None

    for _ in range(N_EPOCHS):
        for (x, y) in dataset:
            a = a if a != None and a.shape[0] == x.shape[0] else None
            loss, _, a = train_step(model, optimizer, x, y, a, GRAD_CLIP)
            with writer.as_default():
                tf.summary.scalar('loss', loss, step=step)
                writer.flush()
            if step % SKIP_STEP == 0:
                print(f'global_step/sec: {step / (time.time() - start_time)}')
                print(f'loss = {loss}, step = {step}')
            if step % SAMPLE_STEP == 0:
                output = sample(model, [char_to_ix[c] for c in INPUT], T_X, T_X)
                output = [ix_to_char[idx] for idx in output]
                print(''.join(output))
                print('\n')
            step += 1

    end_time = time.time()
    print(f'total time: {end_time - start_time}s')
    writer.close()


if __name__ == '__main__':
    main()

# %%
