# %%
import random
import numpy as np
import tensorflow as tf

from babel.dates import format_date
from datetime import datetime, date, timedelta
from tensorflow import keras
from tensorflow.keras import layers
from typing import Generator, Dict, List, Set, Tuple

FORMATS = [
    'short',
    'medium',
    'long',
    'full',
    'full',
    'full',
    'full',
    'full',
    'full',
    'full',
    'full',
    'full',
    'full',
    'd MMM YYY',
    'd MMMM YYY',
    'dd MMM YYY',
    'd MMM, YYY',
    'd MMMM, YYY',
    'dd, MMM YYY',
    'd MM YY',
    'd MMMM YYY',
    'MMMM d YYY',
    'MMMM d, YYY',
    'dd.MM.YY',
]

LOCALE = 'en_US'
EXTRA_VOCAB = ['<unk>', '<pad>']


## dataset


def build_data(
    count: int,
) -> Tuple[List[Tuple[str, str]], Dict[str, int], Dict[str, int]]:
    data = []
    source_vocab: Set[str] = set()
    target_vocab: Set[str] = set()
    for source, target in gen_data(count):
        data.append((source, target))
        source_vocab.update(tuple(source))
        target_vocab.update(tuple(target))
    source_map = dict(
        [(ch, idx) for idx, ch in enumerate(sorted(source_vocab) + EXTRA_VOCAB)]
    )
    target_map = dict([(ch, idx) for idx, ch in enumerate(sorted(target_vocab))])
    return data, source_map, target_map


def build_dataset(
    data: List[Tuple[str, str]],
    input_vocab: Dict[str, int],
    input_width: int,
    output_vocab: Dict[str, int],
    output_width: int,
) -> tf.data.Dataset:
    input_dim = len(input_vocab)
    output_dim = len(output_vocab)
    xs = []
    ys = []
    for x, y in data:
        xs.append(vectorize_string(x, input_vocab, input_width))
        ys.append(vectorize_string(y, output_vocab, output_width))
    ds = tf.data.Dataset.from_tensor_slices((xs, ys))
    ds = ds.map(
        lambda x, y: (
            tf.one_hot(x, input_dim, dtype=tf.int32),
            tf.one_hot(y, output_dim, dtype=tf.int32),
        ),
    )
    return ds


def gen_data(count: int) -> Generator[Tuple[str, str], None, None]:
    now = datetime.now()
    for _ in range(count):
        offset = random.randint(-50 * 365, 50 * 365)
        dt = now + timedelta(days=offset)
        fmt_dt = format_date(dt, format=random.choice(FORMATS), locale=LOCALE)
        fmt_dt = fmt_dt.lower()
        fmt_dt = fmt_dt.replace(',', '')
        iso_dt = dt.isoformat()
        yield fmt_dt, iso_dt


def vectorize_string(input: str, vocab: Dict[str, int], length: int) -> List[int]:
    input = input[:length] if len(input) > length else input
    unk = vocab.get('<unk>', 0)
    indices = [vocab.get(ch, unk) for ch in input]
    if len(input) < length:
        indices += [vocab['<pad>']] * (length - len(input))
    return indices


## model


def build_model(
    input_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
    n_a: int,
    state_dim: int,
) -> keras.Model:
    t_x, _ = input_shape
    t_y, n_y = output_shape
    x = layers.Input(shape=input_shape, name='inputs')
    s0 = layers.Input(shape=(state_dim,), name='s0')
    c0 = layers.Input(shape=(state_dim,), name='c0')

    context_layers = {
        's_prev_repeat': layers.RepeatVector(t_x),
        'concatenate': layers.Concatenate(axis=-1),
        'dense_1': layers.Dense(units=10, activation='tanh'),
        'dense_2': layers.Dense(units=1, activation='relu'),
        'softmax': layers.Activation('softmax'),
        'dot': layers.Dot(axes=1),
    }
    pre_atten = layers.Bidirectional(
        layers.LSTM(units=n_a, return_sequences=True), name='pre-atten'
    )
    post_atten_cell = layers.LSTM(
        units=state_dim, return_state=True, name='post_atten_cell'
    )
    output_layer = layers.Dense(units=n_y, activation='softmax', name='output_cell')

    s = s0
    c = c0
    outputs = []
    a = pre_atten(x)
    for _ in range(t_y):
        context = compute_context(a, s, context_layers)
        s, _, c = post_atten_cell(
            context, initial_state=[s, c] if s is not None else None
        )
        output = output_layer(s)
        outputs.append(output)
    outputs = keras.backend.stack(outputs, axis=1)

    return keras.Model(inputs=[x, s0, c0], outputs=outputs)


def compute_context(a, s_prev, shared_layers):
    """
    Computes one step of attention

    Arguments:
    a - all hidden states of the pre-attention layer (m, t_x, 2*n_a)
    s_prev - previous hidden state of the post-attention layer (m, n_s)

    Returns:
    context - a dot product of the attention weights and the hidden states of pre-atten layer
    """
    # copy 's_prev 't_x times
    s_prev = shared_layers['s_prev_repeat'](s_prev)
    # concatenate 's_prev and 'a
    concat = shared_layers['concatenate']([s_prev, a])
    # combine 's_prev and 'a using simple nn, which learns the function to output e(t, t')
    energies = shared_layers['dense_1'](concat)
    energies = shared_layers['dense_2'](energies)
    # compute attention weights
    alphas = shared_layers['softmax'](energies)
    # compute context
    context = shared_layers['dot']([alphas, a])
    return context


## main


def sample(
    samples: List[str],
    model: keras.Model,
    input_vocab: Dict[str, int],
    input_width: int,
    output_vocab: Dict[str, int],
    state_dim: int,
):
    output_vocab_inv = dict([(v, k) for k, v in output_vocab.items()])
    xs = []
    for sample in samples:
        x = vectorize_string(sample, input_vocab, input_width)
        x = keras.utils.to_categorical(x, num_classes=len(input_vocab))
        xs.append(x)
    xs = np.array(xs)
    s0 = np.zeros((len(xs), state_dim))
    c0 = np.zeros((len(xs), state_dim))
    preds = model.predict([xs, s0, c0])
    preds = np.argmax(preds, axis=-1)
    for source, pred in zip(samples, preds):
        output = [output_vocab_inv[int(idx)] for idx in pred]
        print(source)
        print(''.join(output))
        print('')


def main():
    num_epochs = 500
    early_stop_patience = 20
    batch_size = 128
    n_a = 32
    n_s = 64
    t_x = 30
    t_y = 10

    print('Loading dataset ...')
    data, input_vocab, output_vocab = build_data(10000)
    ds = build_dataset(data, input_vocab, t_x, output_vocab, t_y)
    shuffled_ds = ds.shuffle(1000)
    train_ds = shuffled_ds.take(int(len(ds) * 0.8)).batch(batch_size, drop_remainder=True)
    val_ds = shuffled_ds.skip(int(len(ds) * 0.8)).batch(batch_size, drop_remainder=True)

    print('Building model ...')
    model = build_model(
        input_shape=(t_x, len(input_vocab)),
        output_shape=(t_y, len(output_vocab)),
        n_a=n_a,
        state_dim=n_s,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    # model.summary()

    print('Training model ...')
    s0 = tf.zeros((batch_size, n_s))
    c0 = tf.zeros((batch_size, n_s))
    train_ds = train_ds.map(lambda x, y: ((x, s0, c0), y))
    val_ds = val_ds.map(lambda x, y: ((x, s0, c0), y))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss', patience=early_stop_patience)
    ]
    _ = model.fit(train_ds, epochs=num_epochs, callbacks=callbacks)
    val_perf = model.evaluate(val_ds)
    performance = dict(zip(model.metrics_names, val_perf))
    print(f'Performance: {performance}')

    samples = [
        '3 May 1979',
        '5 April 09',
        '21th of August 2016',
        'Tue 10 Jul 2007',
        'Saturday May 9 2018',
        'March 3 2001',
        'March 3rd 2001',
        '1 March 2001',
    ]
    sample(samples, model, input_vocab, t_x, output_vocab, n_s)


if __name__ == '__main__':
    main()

# %%
