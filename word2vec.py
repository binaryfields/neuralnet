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


# hyperparams
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000


# model
class Word2Vec(object):
    def __init__(self, vocab_size, embed_size, num_sampled):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.embed_matrix = tf.Variable(
            tf.random.uniform([vocab_size, embed_size]), name='embed_matrix'
        )
        self.nce_weights = tf.Variable(
            tf.random.truncated_normal([vocab_size, embed_size], stddev=1.0 / (embed_size ** 0.5)),
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


@tf.function
def train_step(optimizer, model, center_words, target_words):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(center_words, target_words)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


def gen():
    yield from word2vec_utils.batch_gen(
        './datasets/text8.zip', VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD
    )


def main():
    # dataset
    print('loading data ...')
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(tf.int32, tf.int32),
        output_shapes=(tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])),
    )

    # graph
    optimizer = tf.optimizers.SGD(LEARNING_RATE)
    model = Word2Vec(VOCAB_SIZE, EMBED_SIZE, NUM_SAMPLED)

    # train
    print('training model ...')
    writer = tf.summary.create_file_writer(
        'graphs/word2vec/lr' + str(optimizer.learning_rate.numpy())
    )
    start_time = time.time()
    step = 0
    total_loss = 0.0
    for center_words, target_words in dataset:
        if step > NUM_TRAIN_STEPS:
            break
        loss = train_step(optimizer, model, center_words, target_words)
        with writer.as_default():
            tf.summary.scalar('loss', loss, step=step)
        total_loss += loss
        if (step + 1) % SKIP_STEP == 0:
            avg_loss = total_loss / SKIP_STEP
            print(f'loss at step {step}: {avg_loss}')
            total_loss = 0.0
        step += 1
        writer.flush()
    end_time = time.time()
    print(f'total time: {end_time - start_time}s')
    writer.close()


if __name__ == '__main__':
    main()


# %%
