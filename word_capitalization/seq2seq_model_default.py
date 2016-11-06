import tensorflow as tf
from tensorflow.python.ops import rnn_cell


'''
Notes: embedding_attention_seq2seq does NOT use bidirectional encoder, but just rnn_unit

'''
class Model():
    def __init__(self, rnn_unit, source_vocab_size, target_vocab_size, rnn_size, rnn_num_layers, embedding_size,
                 encoder_size, decoder_size,
                 infer=False):
        '''

        :param rnn_unit:
        :param source_vocab_size:
        :param target_vocab_size:
        :param rnn_size:
        :param rnn_num_layers:
        :param embedding_size:
        :param encoder_size: Length of longest sentence passed to encoder.
        :param decoder_size: Length of longest sentence passed to decoder (containing 'GO' and 'EOS' symbols)
        :param infer:
        '''

        if rnn_unit == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif rnn_unit == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif rnn_unit == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(rnn_unit))

        with tf.name_scope('inputs'):
            self.encoder_inputs = []
            for i in range(encoder_size):
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="encoder{0}".format(i)))

            self.decoder_inputs = []
            self.target_weights = []
            self.targets = []
            for i in range(decoder_size):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="decoder{0}".format(i)))

                self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                          name="target_weights{0}".format(i)))

                self.targets.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="targets{0}".format(i)))

            batch_size = tf.shape(self.encoder_inputs)[0]

            cell = cell_fn(rnn_size)

            decoder_outputs, self.state = tf.nn.seq2seq.embedding_attention_seq2seq(self.encoder_inputs,
                                                                                    self.decoder_inputs,
                                                                                    cell,
                                                                                    source_vocab_size,
                                                                                    target_vocab_size,
                                                                                    embedding_size,
                                                                                    feed_previous=False)

            # decoder_outputs is a list (with length decoder_size) and its items are of dimension [batch_size x
            # num_decoder_symbols]


            loss = tf.nn.seq2seq.sequence_loss_by_example(decoder_outputs, self.targets,
                                                          self.target_weights)
            self.cost = tf.reduce_mean(loss)  # LM divided also by max_seq_len

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.cost)

            with tf.name_scope('accuracy'):
                argmax_probs = tf.cast(tf.argmax(decoder_outputs, dimension=2), tf.int32)

                correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)

                correct_pred_masked = correct_pred * self.target_weights

                self.accuracy = tf.reduce_sum(correct_pred_masked) / tf.cast(tf.reduce_sum(self.target_weights),
                                                                             tf.float32)
                tf.scalar_summary('accuracy', self.accuracy)

            self.summaries = tf.merge_all_summaries()
