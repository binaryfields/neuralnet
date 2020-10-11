# %%
# word2vec
#
# - eager execution
# - input dataset from generator
# - custom model layers using tf.nn function
# - custom training loop

import numpy as np
import tensorflow as tf
import time

import word2vec_utils


VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
N_EPOCHS = 50000
NUM_SAMPLED = 64  # number of negative examples to sample
LEARNING_RATE = 1.0
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000


class Word2Vec(object):
    def __init__(self, vocab_size, embed_size, num_sampled):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.embed_matrix = tf.Variable(
            tf.random.uniform([vocab_size, embed_size]), name='embed_matrix'
        )
        self.nce_weights = tf.Variable(
            tf.random.truncated_normal(
                [vocab_size, embed_size], stddev=1.0 / (embed_size ** 0.5)
            ),
            name='nce_weights',
        )
        self.nce_bias = tf.Variable(tf.zeros([vocab_size]), 'nce_bias')
        self.trainable_weights = [self.embed_matrix, self.nce_weights, self.nce_bias]

    def compute_loss(self, center_words, target_words):
        embeddings = tf.nn.embedding_lookup(self.embed_matrix, center_words, name='embed')
        loss = tf.nn.nce_loss(
            weights=self.nce_weights,
            biases=self.nce_bias,
            labels=target_words,
            inputs=embeddings,
            num_sampled=self.num_sampled,
            num_classes=self.vocab_size,
            name='loss',
        )
        loss = tf.reduce_mean(loss)
        return loss


def generate_dataset():
    yield from word2vec_utils.batch_gen(
        './data/text8.zip', VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD
    )


@tf.function
def train_step(optimizer, model, center_words, target_words):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(center_words, target_words)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


def main():
    # dataset
    print('Loading data ...')
    ds = tf.data.Dataset.from_generator(
        generator=generate_dataset,
        output_types=(tf.int32, tf.int32),
        output_shapes=(tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])),
    )

    # model
    model = Word2Vec(VOCAB_SIZE, EMBED_SIZE, NUM_SAMPLED)

    # train
    print('Training model ...')
    optimizer = tf.optimizers.SGD(LEARNING_RATE)
    writer = tf.summary.create_file_writer(
        'graphs/word2vec/lr' + str(optimizer.learning_rate.numpy())
    )
    skip_step = 100
    step = 0
    start_time = time.time()
    for center_words, target_words in ds:
        if step > N_EPOCHS:
            break
        loss = train_step(optimizer, model, center_words, target_words)
        step += 1
        if step % skip_step == 0 or step == N_EPOCHS:
            print(f'{step} - loss: {(loss/step):.4f}')
        with writer.as_default():
            tf.summary.scalar('loss', loss, step=step)
            writer.flush()
    end_time = time.time()
    print(f'Training time: {end_time - start_time}s')
    writer.close()


if __name__ == '__main__':
    main()
