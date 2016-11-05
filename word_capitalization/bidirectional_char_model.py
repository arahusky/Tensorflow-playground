import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


class Model():
    def __init__(self, char_vocab_size, word_vocab_size, char_embedding_size, word_embedding_size, char_rnn_size,
                 rnn_num_layers=1, rnn_unit='gru'):
        '''
        Creates new model for classifying words in sentence as either starting with uppercase or starting with lowercase.

        Model consists of two logical parts. First one creates word embeddings both from its characters (character-level word embeddings) and
        whole words used. Second one feeds this embeddings into bidirectional RNN, whose task is to determine, whether given word (fed as concatenation
        of described embeddings) should start with lowercase (0) or uppercase (1). Both parts are trained jointly
        with backpropagation.

        Note that word embeddings used in the bidirectional RNN are of size word_embedding_size + 2 * char_rnn_size.

        :param char_vocab_size: size of character vocabulary used in the project (any character with index beyond this size will cause an error)
        :param word_vocab_size: size of word vocabulary used in the project (any word with index beyond this size will cause an error)
        :param char_embedding_size: size of embedding vector used for each character (set to 0 when working only with words)
        :param word_embedding_size: size of embedding vector used for each word (set to 0 when working only with characters)
        :param char_rnn_size: size of RNN unit used for creating character level word embeddings (size of char-level word embedding = 2 * char_rnn_size)
        :param rnn_num_layers (optional): number of layers of bidirectional RNN used for classifying whether word starts with uppercase
        :param rnn_unit (optional): what RNN unit to use (possibilities: rnn, gru, lstm), default: gru
        '''

        if word_embedding_size == 0 and char_embedding_size == 0:
            raise ValueError('Both word_embedding_size and char_embedding_size may not be 0.')

        if rnn_unit == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif rnn_unit == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif rnn_unit == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(rnn_unit))

        with tf.name_scope('inputs'):
            '''
            There are several inputs that should be described:
                - self.input_data is [batch_size, max_sentence_words] array that contains for each sample (sentence)
                its words represented by integer indices. These indices point to self.input_words,
                which is described later. Note that these indices are batch-specific, which means that one
                word (e.g. car) may have different index in different batches. To convert this batch-specific word
                index to global one, we use mapping (embedding) stored in self.input_data_mapping. Finally, the length of
                each sample stored in self.input_data is stored in self.input_data_lengths
                - self.input_words is [batch_unique_word_count, batch_max_chars_per_word] array that contains each
                word represented as a sequence of its characters. These characters are also represented by indices,
                which are in this case global (i.e. each character has the same index in all batches). Length of each word
                (=number of relevant characters in this array) is stored in self.input_words_lengths
            '''
            self.input_data = tf.placeholder(tf.int32, [None, None])  # [batch_size, max_sentence_words]
            self.input_data_lengths = tf.placeholder(tf.int32,
                                                     [None])  # [batch_size] - how many words are in each batch item

            # this placeholder stores mapping (embedding) that translates word indices used in self.input_data to
            # indices of whole dataset
            self.input_data_mapping = tf.placeholder(tf.int32, [None, 1])
            self.input_words = tf.placeholder(tf.int32, [None,
                                                         None])  # [unique_words_count, max_word_chars] - matrix (set) containing all words used in current batch
            self.input_words_lengths = tf.placeholder(tf.int32, [
                None])  # [max_count_words] - how many chars are in each self.input_words

            self.targets = tf.placeholder(tf.int32, [None, None])  # [batch_size, max_sentence_words]
            self.keep_prob = tf.placeholder(tf.float32)  # dropout keep probability

            batch_size = tf.shape(self.input_data)[0]
            max_sentence_words = tf.shape(self.input_data)[1]
            max_word_chars = tf.shape(self.input_words)[1]

        # if we should consider character-level word embeddings
        if char_embedding_size > 0:
            with tf.name_scope('char-embedding'):
                '''
                Create char-level word embeddings for all words used, i.e. for all rows in self.input_words. Than use these embeddings for input_data.
                Character level embeddings are created as described in http://arxiv.org/pdf/1508.02096v2.pdf.
                '''
                self.char_embedding = tf.get_variable("char_embedding", [char_vocab_size, char_embedding_size])
                word_inputs = tf.nn.embedding_lookup(self.char_embedding,
                                                     self.input_words)  # inputs is now of dimension [unique_words_count, max_word_chars, char_embedding_size]
                fw_emb_cell = cell_fn(char_rnn_size)
                bw_emb_cell = cell_fn(char_rnn_size)

                _, output_states = rnn.bidirectional_dynamic_rnn(fw_emb_cell, bw_emb_cell, word_inputs,
                                                                 sequence_length=tf.cast(self.input_words_lengths,
                                                                                         tf.int64),
                                                                 dtype=tf.float32)

                output_state_fw, output_state_bw = output_states  # the forward and the backward final states of bidirectional rnn, each of dimension [ unique_words_count, char_rnn_size]
                char_level_word_embedding_matrix = tf.concat(1,
                                                             output_states)  # [ unique_words_count, 2 * char_rnn_size] - word embedding for each word in batch

                print('char_level_word_embedding_matrix: {}'.format(char_level_word_embedding_matrix.get_shape()))
                input_data_with_char_level_word_embeddings = tf.nn.embedding_lookup(char_level_word_embedding_matrix,
                                                                                    self.input_data)  # [batch_size,
                # max_sentence_words, 2 * char_rnn_size]

        # if we should consider word-level word embeddings
        if word_embedding_size > 0:
            input_data_mapped_for_whole_dataset = tf.squeeze(tf.nn.embedding_lookup(self.input_data_mapping,
                                                                                self.input_data), squeeze_dims=[2])
            print('input_data_mapped_for_whole_dataset: {}'.format(input_data_mapped_for_whole_dataset.get_shape()))
            word_embedding_matrix = tf.get_variable("word_embedding", [word_vocab_size, word_embedding_size])
            input_data_with_word_level_word_embeddings = tf.nn.embedding_lookup(word_embedding_matrix,
                                                                                input_data_mapped_for_whole_dataset)  # [batch_size, max_sentence_words, word_embedding_size]

        # print('input_data_with_word_level_word_embeddings: {}'.format(
        #     input_data_with_word_level_word_embeddings.get_shape()))
        # print(input_data_with_char_level_word_embeddings.get_shape())
        word_level_embedding_size = 0
        if char_embedding_size > 0 and word_embedding_size > 0:
            word_level_embedding_size = word_embedding_size + 2 * char_rnn_size
            input_data_with_embeddding = tf.concat(2, [input_data_with_word_level_word_embeddings,
                                                       input_data_with_char_level_word_embeddings])  # [batch_size, max_sentence_words, 2 * config.word_embedding_
        elif char_embedding_size > 0:
            word_level_embedding_size = 2 * char_rnn_size
            input_data_with_embeddding = input_data_with_char_level_word_embeddings
        else:  # word_embedding_size > 0
            word_level_embedding_size = word_embedding_size
            input_data_with_embeddding = input_data_with_word_level_word_embeddings

        input_data_with_embedding_dropout = tf.nn.dropout(input_data_with_embeddding, keep_prob=self.keep_prob)

        with tf.name_scope('RNN_cells'):
            fw_cell = cell_fn(word_level_embedding_size)  # forward direction cell
            bw_cell = cell_fn(word_level_embedding_size)  # backward direction cell

            self.fw_cell = rnn_cell.MultiRNNCell([fw_cell] * rnn_num_layers)
            self.bw_cell = rnn_cell.MultiRNNCell([bw_cell] * rnn_num_layers)

        self.outputs, _ = rnn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, input_data_with_embedding_dropout,
                                                        sequence_length=tf.cast(self.input_data_lengths, tf.int64),
                                                        dtype=tf.float32)

        '''
        Outputs contains two tensors of dimension [batch_size, max_sentence_words, word_level_embedding_size], where first tensor holds output of forward- and second tensor holds output of backward RNN unit.
        Outputs of forward and backward unit are concatenated (tf.concat(2, self.outputs)), which results in a new tensor with dimension [batch_size, max_sentence_words, 2*word_level_embedding_size].
        Before performing softmax in each time frame, we reshape this to [-1, 2*word_level_embedding_size].
        Elements of reshaped tensor are in following order: [output of 1.batch & 1.unit, output of 1.batch & 2.unit, ... , output of 1.batch & config.seq_length-unit, output of 2.batch & 1.unit, ...]
        '''
        self.output = output = tf.reshape(tf.concat(2, self.outputs), [-1, 2 * word_level_embedding_size])

        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable("softmax_w", [2 * word_level_embedding_size, 2])
            softmax_b = tf.get_variable("softmax_b", [2])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        with tf.name_scope('cost'):
            '''
            Outputs is of size [batch_size, max_seq_len, 2*rnn_size] but with the last elements being zero vectors for sequences shorter than the maximum sequence length.
            We apply softmax_cross_entropy on outputs and then compute mean of them, while not considering values behind each sequence length.
            To filter out these 'behind values', we compute a binary mask having 1's for valid and 0's for invalid positions.
            '''
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                tf.reshape(self.logits, [batch_size, max_sentence_words, 2]), self.targets)

            with tf.name_scope('mask'):
                # http://stackoverflow.com/questions/34128104/tensorflow-creating-mask-of-varied-lengths
                # create a matrix with len(self.inputs_lengths) rows, where each row contains corresponding length repeated max_sequence_length times
                lengths_transposed = tf.expand_dims(self.input_data_lengths, 1)
                lengths_tiled = tf.tile(lengths_transposed, [1, max_sentence_words])

                # create [len(self.inputs_lengths), max_sequence_length] matrix, where each row contains [0, 1, ..., max_sequence_length]
                range = tf.range(0, max_sentence_words, 1)
                range_row = tf.expand_dims(range, 0)
                range_tiled = tf.tile(range_row, [batch_size, 1])

                # use the logical less-than operation to produce boolean mask, which is then converted to float32 binary mask
                self.mask = tf.cast(tf.less(range_tiled, lengths_tiled), tf.float32)

            cross_entropy_masked = cross_entropy * self.mask
            self.cost = tf.reduce_sum(cross_entropy_masked) / tf.cast(tf.reduce_sum(self.input_data_lengths),
                                                                      tf.float32)

            # TODO initial learning rate + weight decay
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
            # TODO gradient clip?

        # accuracy
        with tf.name_scope('accuracy'):
            argmax_probs = tf.reshape(tf.cast(tf.argmax(self.probs, dimension=1), tf.int32),
                                      [batch_size, max_sentence_words])
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            correct_pred_masked = correct_pred * self.mask

            self.accuracy = tf.reduce_sum(correct_pred_masked) / tf.cast(tf.reduce_sum(self.input_data_lengths),
                                                                         tf.float32)
            tf.scalar_summary('accuracy', self.accuracy)

        self.summaries = tf.merge_all_summaries()
