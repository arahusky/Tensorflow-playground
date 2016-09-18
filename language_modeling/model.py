from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, config, infer=False):
        self.config = config
        if infer:
            config.batch_size = 1
            config.seq_length = 1

        if config.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif config.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif config.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(config.model))

        cell = cell_fn(config.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * config.num_layers)

        self.input_data = tf.placeholder(tf.int32, [config.batch_size, config.seq_length])
        self.targets = tf.placeholder(tf.int32, [config.batch_size, config.seq_length])
        self.initial_state = cell.zero_state(config.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [config.rnn_size, config.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [config.vocab_size])
            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable("embedding", [config.vocab_size, config.rnn_size])
                inputs = tf.split(1, config.seq_length, tf.nn.embedding_lookup(self.embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            """
            This function takes decoder output at t-1 and returns input for the decoder at t 
            """
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, config.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([config.batch_size * config.seq_length])], # all samples have same weight
                config.vocab_size)
        self.cost = tf.reduce_sum(loss) / config.batch_size / config.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                config.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num=200, prime = '', sampling_type=2):
        """
        Returns text starting with char/word 'prime' word and trying to maximize text's likelihood.
        
        When 'prime' is empty, text starts with a random char/	word from 'words'.
        """
        state = self.cell.zero_state(1, tf.float32).eval()
        
        if prime == '':
            prime = vocab.keys()[np.random.randint(len(vocab.keys()))]
            print('a' + prime)
        print ('Starting with word ' + prime)
        
        x = np.zeros((1, 1))
        x[0, 0] = vocab[prime]
        feed = {self.input_data: x, self.initial_state:state}
        [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            """
            Gets weights of given words and selects one word randomly (with respect to weights = probabilities) (= roullete)
            """
            t = np.cumsum(weights)
            s = np.sum(weights)
            
            # finds position in t, where np.random.rand(1)*s would be inserted to preserve ordering
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = [prime]
        word = prime
        for n in range(num):
            x[0, 0] = vocab[word]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if word == '\n':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = words[sample]
            ret.append(pred)
            word = pred
        return ret
    
        
    def get_embeddings(self, sess):
        return sess.run(self.embedding)
    
    def get_probability(self, sess, sequence, vocab):
        """
        Computes probability of given sequence (list of strings).
        """
        state = self.cell.zero_state(1, tf.float32).eval()
        x = np.zeros((1,1))        
        seq_prob = 1.0
        for (i,word) in enumerate(sequence):
            x[0,0] = vocab[word]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            probs = probs[0] # probs returned from run is 2d array with only one item
            if i != len(sequence)-1:
                next_word = sequence[i+1]
                seq_prob *= probs[vocab[next_word]]
                
        return seq_prob
        
