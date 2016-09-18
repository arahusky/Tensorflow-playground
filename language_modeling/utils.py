import os
import tensorflow as tf
from six.moves import cPickle


def sample(args, model, is_char_model):
    '''
    Generates text having high probability.
    '''
    with open(os.path.join(args.save_dir, 'vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            sampled = model.sample(sess, words, vocab, args.n, args.prime, args.sample)
            if is_char_model:
                print(''.join(sampled))
            else:
                print(' '.join(sampled))


def compute_seq_prob(seq, model, args):
    '''
    Computes probability of provided sequence. Note that this usually makes sense only for comparing several sequences.
    '''
    with open(os.path.join(args.save_dir, 'vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        return model.get_probability(sess, seq, vocab)
