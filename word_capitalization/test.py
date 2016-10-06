import os
import tensorflow as tf
from six.moves import cPickle
import text_loader
from config import Config
import model
import nltk
import numpy as np


def capitalize_words(words, probs):
    res = []
    for (i, word) in enumerate(words):
        if probs[i][0] < probs[i][1]:
            res.append(word[0].upper() + word[1:])
        else:
            res.append(word)

    return res


sentence_to_capitalize = 'a man went to london on tuesday. this city lies in uk, where this peter griffin was born '

config = Config()


with open(os.path.join(text_loader.TextLoader.CACHE_FOLDER, 'vocab.pkl'), 'rb') as f:
    words = cPickle.load(f)

vocab = dict(zip(words, range(len(words))))
config.vocab_size = len(vocab)


words = nltk.word_tokenize(sentence_to_capitalize)
print(words)
x = np.array([vocab[word] for word in words])
x = x.reshape((1,len(words)))
print(sentence_to_capitalize, x, x.shape)

config.seq_length = len(words)
model = model.Model(config, infer=True)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state(Config.save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        feed = {model.input_data: x}  # , model.initial_state: state}
        probs = sess.run([model.probs], feed)
        print(probs)
        print(capitalize_words(nltk.word_tokenize(sentence_to_capitalize), probs[0]))
