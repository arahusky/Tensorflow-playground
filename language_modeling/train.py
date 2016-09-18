# Training (skip if the model is already trained)
import time, os
import tensorflow as tf
from text_loader import TextLoader
from config import Config
from model import Model
from six.moves import cPickle


def train (is_char_model = True):
    config = Config()

    data_loader = TextLoader(config.data_dir, config.batch_size, config.seq_length, is_char_model)
    config.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if config.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(config.init_from), " %s must be a a path" % config.init_from
        assert os.path.isfile(
            os.path.join(config.init_from, "config.pkl")), "config.pkl file does not exist in path %s" % config.init_from
        assert os.path.isfile(os.path.join(config.init_from,
                                           "vocab.pkl")), "vocab.pkl file does not exist in path %s" % config.init_from
        ckpt = tf.train.get_checkpoint_state(config.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(config.init_from, 'config.pkl'), 'rb') as f:
            saved_model_config = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_config)[checkme] == vars(config)[
                checkme], "Command line argument and saved model disagree on '%s' " % checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(config.init_from, 'vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = cPickle.load(f)
        assert saved_words == data_loader.words, "Data and loaded model disagreee on word set!"
        assert saved_vocab == data_loader.vocab, "Data and loaded model disagreee on dictionary mappings!"

    with open(os.path.join(config.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(config, f)
    with open(os.path.join(config.save_dir, 'vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

    model = Model(config)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        # restore model
        if config.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(config.num_epochs):
            sess.run(tf.assign(model.lr, config.learning_rate * (config.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()

                if (e * data_loader.num_batches + b) % config.save_every == 0 \
                        or (
                                e == config.num_epochs - 1 and b == data_loader.num_batches - 1):  # save for the last result
                    checkpoint_path = os.path.join(config.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.num_batches + b,
                              config.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))

if __name__ == '__main__':
    train()
