import os
import tensorflow as tf
from six.moves import cPickle
import text_loader_bidi_word_model
from config import Config
import bidirectional_word_model
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

new_config = Config()
with open(os.path.join(new_config.save_dir, 'config.pkl'), 'rb') as f:
    config = cPickle.load(f)

sentence_to_capitalize = 'a man went to london on tuesday. this city lies in uk, where this peter griffin was born '

loader = text_loader_bidi_word_model.TextLoader(config)

vocab = loader.vocab
config.vocab_size = len(vocab)


words = nltk.word_tokenize(sentence_to_capitalize)
x = np.array([vocab[word] for word in words])
x = x.reshape((1,len(words)))
print(sentence_to_capitalize, x, x.shape)

model = bidirectional_word_model.Model(config, infer=True)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state(Config.save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        feed = {}
        feed[model.input_data] = x
        feed[model.inputs_lengths] = [x.shape[1]]
        probs = sess.run([model.probs], feed)
        print(probs)
        print(capitalize_words(nltk.word_tokenize(sentence_to_capitalize), probs[0]))

        # eval test set accuracy
        test_batches_x, test_batches_lengths, test_batches_y = loader.get_test_set()
        feed = {}
        feed[model.input_data] = test_batches_x
        feed[model.inputs_lengths] = test_batches_lengths
        feed[model.targets] = test_batches_y
        print('Test set accuracy: ' + str(sess.run(model.accuracy, feed)))
