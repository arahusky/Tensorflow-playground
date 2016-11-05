import time, os
import tensorflow as tf
from config import Config
from six.moves import cPickle
import text_loader_for_bidi_char_model
import bidirectional_char_model, config
import numpy as np
import datetime

experiment_name = 'char200_word100_UNK_model'

def train():
    config = Config()

    print('Loading dataset')
    data_loader = text_loader_for_bidi_char_model.TextLoader(config)

    # check compatibility if training is continued from previously saved model
    if config.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(config.init_from), " %s must be a a path" % config.init_from
        assert os.path.isfile(
            os.path.join(config.init_from,
                         "config.pkl")), "config.pkl file does not exist in path %s" % config.init_from
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
            saved_vocab = cPickle.load(f)
        assert saved_vocab == data_loader.char_vocab, "Data and loaded model disagreee on dictionary mappings!"

    with open(os.path.join(config.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(config, f)
    with open(os.path.join(config.save_dir, 'vocab.pkl'), 'wb') as f:
        cPickle.dump(data_loader.char_vocab, f)

    print('Creating model')
    model = bidirectional_char_model.Model(char_vocab_size = data_loader.char_vocab_size,
                        word_vocab_size = data_loader.word_vocab_size,
                        char_embedding_size = config.char_embedding_size,
                        word_embedding_size = config.word_embedding_size,
                        char_rnn_size = config.char_rnn_size,
                        rnn_num_layers = config.num_layers,
                        rnn_unit = config.model)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    summary_writer = tf.train.SummaryWriter('{}/{}-{}'.format(config.log_dir, timestamp, experiment_name), graph=tf.get_default_graph())
    thread_to_use = 8

    print('Starting training')
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=thread_to_use,
                            intra_op_parallelism_threads=thread_to_use)) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        # restore model
        if config.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for e in range(config.num_epochs):
            # sess.run(tf.assign(model.lr, config.learning_rate * (config.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()

                input_data, input_data_lengths, batch_targets, batch_word_mapping, batch_input_words, batch_input_words_lengths \
                    = data_loader.next_batch()
                middle = time.time()
                # print('x.shape' + str(input_data.shape) + 'y,shape' + str(batch_targets.shape))
                feed = {model.input_data: input_data,
                        model.input_data_lengths: input_data_lengths,
                        model.targets: batch_targets,
                        model.input_data_mapping: np.array(batch_word_mapping.values()).reshape([len(batch_word_mapping.keys()), 1]),
                        model.input_words: batch_input_words,
                        model.input_words_lengths: batch_input_words_lengths,
                        model.keep_prob: config.keep_prob}
                # train_loss, acc, _ = sess.run([model.cost, model.accuracy, model.optimizer], feed)
                train_loss, acc, _ = sess.run([model.cost, model.accuracy, model.optimizer], feed)
                end = time.time()

                if (e * data_loader.num_batches + b) % config.save_every == 0 \
                        or (
                                        e == config.num_epochs - 1 and b == data_loader.num_batches - 1):  # save for the last result
                    checkpoint_path = os.path.join(config.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                if ((e * data_loader.num_batches + b) % 100 == 0):  # print test accuracy
                    test_input_data, test_input_data_lengths, test_targets, test_word_mapping, test_input_words, \
                    test_input_words_lengths = data_loader.get_test_set()
                    # print('input_words.shape' + str(test_input_data.shape))
                    feed = {model.input_data: test_input_data,
                            model.input_data_lengths: test_input_data_lengths,
                            model.targets: test_targets,
                            model.input_data_mapping: np.array(test_word_mapping.values()).reshape(
                                [len(test_word_mapping.keys()), 1]),
                            model.input_words: test_input_words,
                            model.input_words_lengths: test_input_words_lengths,
                            model.keep_prob: 1.0}

                    summary, acc = sess.run([model.summaries, model.accuracy], feed)
                    print('Test accuracy: ' + str(acc))
                    summary_writer.add_summary(summary, e * data_loader.num_batches + b)

                msg = ("{}/{} (epoch {}), train_loss = {:.3f}, train_acc = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.num_batches + b,
                              config.num_epochs * data_loader.num_batches,
                              e, train_loss, acc, end - start))

                with open('output_both.txt', 'a') as writer:
                    writer.write(msg + "\n")

                print(msg)
                print("Batch prepare took: {}".format(middle - start))


if __name__ == '__main__':
    train()
