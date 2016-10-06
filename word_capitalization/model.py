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

        self.input_data = tf.placeholder(tf.int32, [config.batch_size, config.seq_length])
        self.targets = tf.placeholder(tf.int32, [config.batch_size, config.seq_length])
        self.fw_initial_state = self.fw_cell.zero_state(config.batch_size, tf.float32)
        self.bw_initial_state = self.bw_cell.zero_state(config.batch_size, tf.float32)

        print('input.data.shape' + str(self.input_data.get_shape()))
        print('targets.shape' + str(self.targets.get_shape()))

        with tf.variable_scope('bidinn'):
            softmax_w = tf.get_variable("softmax_w", [ 2 * config.rnn_size, 2])
            softmax_b = tf.get_variable("softmax_b", [2])
            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable("embedding", [config.vocab_size, config.rnn_size])
                inputs = tf.split(1, config.seq_length, tf.nn.embedding_lookup(self.embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        print('inputs.get_shape()' + str(inputs[0].get_shape()) + ":" + str(len(inputs)))
        outputs, output_state_fw, output_state_bw = rnn.bidirectional_rnn(self.fw_cell, self.bw_cell, inputs, self.fw_initial_state, self.bw_initial_state)

        print('outputs.get_shape(' + str(len(outputs)) + ":" + str(outputs[0].get_shape()))
        # reshape output to [-1, 2*config.rnn_size] to allow matrix multiply, elements of resulting tensor are in a following order:
        # [output of 1.batch & 1.unit, output of 1.batch & 2.unit, ... , output of 1.batch & config.seq_length-unit, output of 2.batch & 1.unit, ...]
        output = tf.reshape(tf.concat(1, outputs), [-1, 2 * config.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b # not reshaped
        self.probs = tf.nn.softmax(self.logits)

        print('output.get_shape(' + str(output.get_shape()))
        print('self.logits.get_shape()' + str(self.logits.get_shape()))
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.reshape(self.logits, [config.batch_size, config.seq_length, 2]), self.targets))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.cost)
        # TODO gradient clip?

        # accuracy
        print('self.probs.get_shape()' + str(self.probs.get_shape()))
        argmax_probs = tf.reshape(tf.cast(tf.argmax(self.probs, dimension=1),tf.int32), [config.batch_size, config.seq_length])
        print('argmax_probs_shape:' + str(argmax_probs.get_shape()) + "targets.shape" + str(self.targets.get_shape()))
        correct_pred = tf.equal(argmax_probs, self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))