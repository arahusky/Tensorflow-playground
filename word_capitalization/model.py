from tensorflow.python.ops import rnn, rnn_cell

import tensorflow as tf
import numpy as np


class Model():
    def __init__(self, config, infer=False):
        self.config = config
        if infer:
            config.batch_size = 1
            # config.seq_length = 20

        if config.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif config.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif config.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(config.model))

        fw_cell = cell_fn(config.rnn_size) # forward direction cell
        bw_cell = cell_fn(config.rnn_size) # backward direction cell

        self.fw_cell = rnn_cell.MultiRNNCell([fw_cell] * config.num_layers)
        self.bw_cell = rnn_cell.MultiRNNCell([bw_cell] * config.num_layers)

        # self.input_data = tf.placeholder(tf.int32, [config.batch_size, config.max_length]) # [batch_size, max_length]
        # self.inputs_lengths = tf.placeholder(tf.int64, [config.batch_size]) # [batch_size]
        # self.targets = tf.placeholder(tf.int32, [config.batch_size, config.max_length]) # same as input_data
        self.input_data = tf.placeholder(tf.int32, [None, None]) # [batch_size, max_length]
        self.inputs_lengths = tf.placeholder(tf.int32, [None]) # [batch_size]
        self.targets = tf.placeholder(tf.int32, [None, None]) # same as input_data
        self.fw_initial_state = self.fw_cell.zero_state(config.batch_size, tf.float32)
        self.bw_initial_state = self.bw_cell.zero_state(config.batch_size, tf.float32)

        # print('input.data.shape' + str(self.input_data.get_shape()))
        # print('targets.shape' + str(self.targets.get_shape()))

        with tf.variable_scope('bidinn'):
            softmax_w = tf.get_variable("softmax_w", [ 2 * config.rnn_size, 2])
            softmax_b = tf.get_variable("softmax_b", [2])
            self.embedding = tf.get_variable("embedding", [config.vocab_size, config.rnn_size])
            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
        print('inputs.get_shape()' + str(inputs[0].get_shape()) + ":" + str(inputs.get_shape()))

        # outputs, _ = rnn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, inputs, dtype=tf.float32)
        outputs, _ = rnn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, inputs, sequence_length=tf.cast(self.inputs_lengths, tf.int64),initial_state_fw= self.fw_initial_state, initial_state_bw=self.bw_initial_state)

        print('outputs.get_shape(' + str(len(outputs)) + ":" + str(outputs[0].get_shape()))
        # reshape output to [-1, 2*config.rnn_size] to allow matrix multiply, elements of resulting tensor are in a following order:
        # [output of 1.batch & 1.unit, output of 1.batch & 2.unit, ... , output of 1.batch & config.seq_length-unit, output of 2.batch & 1.unit, ...]
        output = tf.reshape(outputs, [-1, 2 * config.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        print('output.get_shape(' + str(output.get_shape()))
        print('self.logits.get_shape()' + str(self.logits.get_shape()))

        with tf.name_scope('cost'):
            '''
            Outputs is of size [batch_size, max_seq_len, 2*rnn_size] but with the last being zero vectors for sequences shorter than the maximum sequence length.
            We apply softmax_cross_entropy on outputs and then compute mean of them, while not considering values behind each sequence length.
            To filter out these 'behind values', we compute a binary mask having 1's for valid and 0's for invalid positions.
            '''
            print('self.targets.get_shape()' + str(self.targets.get_shape()))
            batch_size = tf.shape(self.targets)[0]
            max_length = tf.shape(self.targets)[1]
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.reshape(self.logits, [batch_size, max_length, 2]), self.targets)

            with tf.name_scope('mask'):
                # http://stackoverflow.com/questions/34128104/tensorflow-creating-mask-of-varied-lengths
                # create a matrix with len(self.inputs_lengths) rows, where each row contains corresponding length repeated max_sequence_length times
                lengths_transposed = tf.expand_dims(self.inputs_lengths, 1)
                lengths_tiled = tf.tile(lengths_transposed, [1, max_length])

                # create [len(self.inputs_lengths), max_sequence_length] matrix, where each row contains [0, 1, ..., max_sequence_length]
                range = tf.range(0, max_length, 1)
                range_row = tf.expand_dims(range, 0)
                range_tiled = tf.tile(range_row, [batch_size, 1])

                # use the logical less than operation to produce boolean mask, which is then converted to float32 binary mask
                self.mask = tf.cast(tf.less(range_tiled, lengths_tiled), tf.float32)

            cross_entropy_masked = cross_entropy * self.mask
            self.cost = tf.reduce_sum(cross_entropy_masked) / tf.cast(tf.reduce_sum(self.inputs_lengths), tf.float32)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.cost)
            # TODO gradient clip?

        # accuracy
        with tf.name_scope('accuracy'):
            print('self.probs.get_shape()' + str(self.probs.get_shape()))
            argmax_probs = tf.reshape(tf.cast(tf.argmax(self.probs, dimension=1),tf.int32), [batch_size, max_length])
            print('argmax_probs_shape:' + str(argmax_probs.get_shape()) + "targets.shape" + str(self.targets.get_shape()))
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            correct_pred_masked = correct_pred * self.mask

            self.accuracy = tf.reduce_sum(correct_pred_masked) / tf.cast(tf.reduce_sum(self.inputs_lengths), tf.float32)