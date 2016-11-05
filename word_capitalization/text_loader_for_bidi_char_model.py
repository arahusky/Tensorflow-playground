# -*- coding: utf-8 -*-

# This class loads, cleans and parses train text into batches.
import collections
import os
import re

import nltk
import numpy as np
from six.moves import cPickle

import config
import utils


class TextLoader():
    '''
    This class provides methods to work with the dataset.
    To load dataset, create new instance of this class with provided data_dir pointing to root folder of dataset.
    '''

    # this folder stores twofiles: vocabulary (single words in order as used in vocabulary) and processed tensor file
    CACHE_FOLDER = config.Config.save_dir

    def __init__(self, config, data_dir='bbc'):
        self.data_dir = data_dir
        self.unknown_symbol = config.unknown_symbol
        self.batch_size = config.batch_size

        train_percentage = config.train_
        valid_percentage = config.validation
        test_percentage = config.test

        vocab_file = os.path.join(self.CACHE_FOLDER, 'vocab.pkl')
        data_file = os.path.join(self.CACHE_FOLDER, 'data.pkl')

        if not (os.path.exists(vocab_file) and os.path.exists(data_file)):
            print('Preprocessing')
            self.preprocess(vocab_file, data_file)
        else:
            print('Loading preprocessed')
            self.load_preprocessed(vocab_file, data_file)
        self.create_batches(train_percentage, valid_percentage, test_percentage)
        self.reset_batch_pointer()

    def load_texts(self):
        text = ''
        topic_folders = os.listdir(self.data_dir)
        for topic_folder in topic_folders:
            if os.path.isdir(os.path.join(self.data_dir, topic_folder)):  # consider only folders
                # print('Loading topic:' + topic_folder)
                topic_texts = os.listdir(os.path.join(self.data_dir, topic_folder))

                for topic_text in topic_texts:
                    # print(topic_text)
                    with open(os.path.join(self.data_dir, topic_folder, topic_text), 'r') as reader:
                        # skip first line (heading)
                        reader.readline()
                        line = reader.readline().decode("ascii", "ignore").encode("ascii")
                        while line:
                            if line.strip() is not '':
                                line = line.strip()
                                # line = re.sub(r'\.([a-z]+)', r'. \1', line)
                                text += " " + line
                            line = reader.readline().decode("ascii", "ignore").encode("ascii")

        return text

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)  # TODO add dot
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def build_vocab(self, words, remove_samples_with_low_occurence = False):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        If remove_samples_with_low_occurence is True, all samples with occurence 1 will be removed (replaced by UNK
        token).
        Returns vocabulary mapping and all appeared words.
        """
        # Build vocabulary
        word_counts = collections.Counter(words)
        # for each word its number of occurence

        if remove_samples_with_low_occurence:
            # print('Removing low occurence words')
            words = [word for word, num_occur in word_counts.most_common() if num_occur > 1]
            # print('Before words: {}, after words: {}'.format(len(word_counts), len(words)))
        else:
            # word_counts.most_common() returns tuples in form (word, num_occurence)
            words = [word for word, num_occur in word_counts.most_common()]

        words = list(sorted(words))
        words.append(self.unknown_symbol)  # symbol for items not seen in dataset

        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(words)}
        return vocabulary

    def preprocess(self, vocab_file, data_file):
        data = self.load_texts()

        # Text cleaning
        # data = self.clean_str(data)

        sentences = nltk.sent_tokenize(data)

        self.max_words = 0

        # [[word_1, word_2], [word1], [word1, word2, word3]]
        sentences_as_seq_of_words = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)

            if len(words) <= config.Config.max_words_in_sentence:
                sentences_as_seq_of_words.append(words)
                if len(words) > self.max_words:
                    self.max_words = len(words)

        # Create labels & convert each word to lower-case version
        self.labels = []
        lower_case_sentences_as_sequence_of_words = []  # [[sentence1_word1, sentence1_word2], [sentence2_word1]]
        for sentence in sentences_as_seq_of_words:
            self.labels.append(np.array(['1' if word[0].isupper() else '0' for word in sentence]))
            lower_case_sentences_as_sequence_of_words.append([word.lower() for word in sentence])

        # [[[char1, char2], [char1,char2, char3]], [[char1, char2]]]
        sentences_as_seq_of_char_arrays = []
        for sentence in lower_case_sentences_as_sequence_of_words:
            new_sentence = []
            for word in sentence:
                characters = list(word)
                new_sentence.append(characters)
            sentences_as_seq_of_char_arrays.append(new_sentence)

        self.word_vocab = self.build_vocab(utils.flatten_list_of_lists(lower_case_sentences_as_sequence_of_words),
                                           remove_samples_with_low_occurence=True)
        self.char_vocab = self.build_vocab(
            utils.flatten_list_of_lists(utils.flatten_list_of_lists(sentences_as_seq_of_char_arrays)))
        words = self.word_vocab.keys()
        characters = self.char_vocab.keys()
        # vocab is a dictionary {"character" : ID}
        # characters is just a list of characters that appeared in the text
        self.word_vocab_size = len(words)
        self.char_vocab_size = len(characters)

        with open(vocab_file, 'wb') as f:
            cPickle.dump((self.word_vocab, self.char_vocab), f)

        # input_data is a list of numpy arrays, where each numpy array is a sequence od word-IDs representing one word
        self.input_data = []
        for sentence in lower_case_sentences_as_sequence_of_words:
            self.input_data.append(np.array([self.word_vocab[word] if word in self.word_vocab else self.word_vocab[
                self.unknown_symbol] for word in sentence]))

        self.input_words = np.zeros((len(words), self.max_words),
                                    dtype=np.int32)  # stores each word in vocabulary as a sequence of its characters
        self.input_words_lengths = np.zeros(len(words), dtype=np.int32)  # stores length for each word in vocabulary
        for word_index, word in enumerate(words):
            self.input_words_lengths[word_index] = len(word)
            for char_index, character in enumerate(word):
                if character in self.char_vocab:
                    self.input_words[word_index][char_index] = self.char_vocab[character]
                else:
                    self.input_words[word_index][char_index] = self.char_vocab[self.unknown_symbol]

        # BBC articles are sorted according to topic, which makes the classic train/test split of this array topic-biased
        permutation = np.random.permutation(len(self.input_data))
        self.input_data = [self.input_data[i] for i in permutation]
        self.labels = [self.labels[i] for i in permutation]

        print('Saving data')
        # Save data
        with open(data_file, 'wb') as f:
            cPickle.dump((self.input_data, self.labels, self.input_words, self.input_words_lengths, self.max_words), f)

    def load_preprocessed(self, vocab_file, data_file):
        with open(vocab_file, 'rb') as f:
            self.word_vocab, self.char_vocab = cPickle.load(f)
        words = self.word_vocab.keys()
        characters = self.char_vocab.keys()
        self.word_vocab_size = len(words)
        self.char_vocab_size = len(characters)
        with open(data_file, 'rb') as f:
            self.input_data, self.labels, self.input_words, self.input_words_lengths, self.max_words = cPickle.load(f)

    def create_batches(self, train_percentage, valid_percentage, test_percentage):
        # TODO rename me
        num_available_batches = int(len(self.input_data) / self.batch_size)

        # counts in means of batches
        self.num_batches = int(round(num_available_batches * train_percentage))  # number of train batches prepared
        num_validation_batches = int(
            round(num_available_batches * valid_percentage))  # number of validation batches prepared
        num_test_batches = num_available_batches - self.num_batches - num_validation_batches  # number of test batches prepared

        # counts in means of samples
        num_train_samples = self.num_batches * self.batch_size
        num_validation_samples = num_validation_batches * self.batch_size
        num_test_samples = num_test_batches * self.batch_size

        if self.num_batches == 0:
            assert False, "Not enough data. Make batch_size smaller."

        print(str(self.num_batches) + ' train batches available')
        print(str(num_validation_batches) + ' validation batches available')
        print(str(num_test_batches) + ' test batches available')
        # cut end part after the last expected batch to ensure that the data can be splitted into batches with no padding
        max_length = num_available_batches * self.batch_size  # how many samples may be prepared
        self.input_data = self.input_data[:max_length]
        self.labels = self.labels[:max_length]

        # split tensor into train, validation and test sets
        self.train_xdata = self.input_data[:num_train_samples]
        self.train_ydata = self.labels[:num_train_samples]

        self.validation_xdata = self.input_data[num_train_samples:num_train_samples + num_validation_samples]
        self.validation_ydata = self.labels[num_train_samples:num_train_samples + num_validation_samples]

        self.test_xdata = self.input_data[num_train_samples + num_validation_samples:]
        self.test_ydata = self.labels[num_train_samples + num_validation_samples:]

        # create numpy array for saving current batch
        self.batch_input_data = np.zeros((self.batch_size, self.max_words), dtype=np.int32)
        self.input_data_lengths = np.zeros(self.batch_size,
                                           dtype=np.int32)  # number of words for each sentence in batch
        self.batch_targets = np.zeros((self.batch_size, self.max_words),
                                      dtype=np.int32)  # contains for each train sentence its word labels

    def next_batch(self):
        '''
        TODO rename description
        Returns next train batch.  Each batch consists of three parts:
            batch_x ; contains self.num_batches of train inputs with variable lengths, note that this array has
             dimensions [self.num_batches, max_length], where max_length is a length of the longest sequence in the training set
            batch_x_length: for each sequence in the batch contains its length
            batch_y: contains labels for batch_x inputs
        '''

        self.batch_input_data.fill(0)

        # determine, what words will be used in current batch
        words_used = set()  # these are actually not words, but just theirs indices (pointing to self.input_words)
        for sample_ind in range(self.batch_size):
            for word_ind in range(len(self.train_xdata[self.pointer + sample_ind])):
                if self.train_xdata[self.pointer + sample_ind][word_ind] not in words_used:
                    words_used.add(self.train_xdata[self.pointer + sample_ind][word_ind])

        batch_word_vocabulary = {x: i for i, x in enumerate(words_used)}
        inverse_batch_word_vocabulary = {value: key for key, value in batch_word_vocabulary.iteritems()}

        max_word_len_in_batch = max(
            [self.input_words_lengths[word] for word in words_used])  # maximum length of word used in the current batch
        batch_input_words = np.zeros((len(words_used), max_word_len_in_batch),
                                     dtype=np.int32)  # for each word in batch its character indices
        batch_input_words_lengths = np.zeros(len(words_used), dtype=np.int32)  # length of each word in batch

        for old_batch_word_index, new_batch_word_index in batch_word_vocabulary.iteritems():
            batch_input_words_lengths[new_batch_word_index] = self.input_words_lengths[old_batch_word_index]
            for word_character_ind in range(batch_input_words_lengths[new_batch_word_index]):
                batch_input_words[new_batch_word_index][word_character_ind] = self.input_words[old_batch_word_index][
                    word_character_ind]

        # now, when we have new vocabulary for the batch, we can prepare batch inputs
        for sample_ind in range(self.batch_size):  # for each sentence in batch
            self.input_data_lengths[sample_ind] = len(self.train_xdata[self.pointer])  # num words in this sentence

            # for each word in sentence
            for word_ind in range(self.input_data_lengths[sample_ind]):
                old_word_index = self.train_xdata[self.pointer][word_ind]
                self.batch_input_data[sample_ind][word_ind] = batch_word_vocabulary[old_word_index]
                self.batch_targets[sample_ind][word_ind] = self.train_ydata[self.pointer][word_ind]

            self.pointer += 1
        return self.batch_input_data, self.input_data_lengths, self.batch_targets, inverse_batch_word_vocabulary, \
               batch_input_words, batch_input_words_lengths

    def get_test_set(self):
        '''
        Returns list of test batches.
        '''

        testdata_samples = len(self.test_xdata)

        # determine, what words are used in test set
        words_used = set()  # these are actually not words, but just theirs indices (pointing to self.input_words)
        for sample_ind in range(testdata_samples):
            for word_ind in range(len(self.test_xdata[sample_ind])):
                if self.test_xdata[sample_ind][word_ind] not in words_used:
                    words_used.add(self.test_xdata[sample_ind][word_ind])

        test_word_vocabulary = {x: i for i, x in enumerate(words_used)}
        inverse_test_word_vocabulary = {value: key for key, value in test_word_vocabulary.iteritems()}

        max_word_len_in_test_set = max(
            [self.input_words_lengths[word] for word in words_used])  # maximum length of word used in the test set
        test_input_words = np.zeros((len(words_used), max_word_len_in_test_set),
                                     dtype=np.int32)  # for each word in batch its character indices
        test_input_words_lengths = np.zeros(len(words_used), dtype=np.int32)  # length of each word in batch

        for old_batch_word_index, new_batch_word_index in test_word_vocabulary.iteritems():
            test_input_words_lengths[new_batch_word_index] = self.input_words_lengths[old_batch_word_index]
            for word_character_ind in range(test_input_words_lengths[new_batch_word_index]):
                test_input_words[new_batch_word_index][word_character_ind] = self.input_words[old_batch_word_index][
                    word_character_ind]

        self.test_input_data = np.zeros((testdata_samples, self.max_words))
        self.test_input_data_lengths = np.zeros(testdata_samples, dtype=np.int32)
        self.test_targets = np.zeros((testdata_samples, self.max_words))
        print('testdata samples', testdata_samples)
        # for each test sentence
        for sample_ind in range(testdata_samples):
            self.test_input_data_lengths[sample_ind] = len(self.test_xdata[sample_ind])

            # for each word in sentence
            for word_ind in range(self.test_input_data_lengths[sample_ind]):
                old_word_index = self.test_xdata[sample_ind][word_ind]
                self.test_input_data[sample_ind][word_ind] = test_word_vocabulary[old_word_index]
                self.test_targets[sample_ind][word_ind] = self.test_ydata[sample_ind][word_ind]

        return self.test_input_data, self.test_input_data_lengths, self.test_targets, inverse_test_word_vocabulary, \
               test_input_words, test_input_words_lengths

    def get_validation_set(self):
        '''
        Returns matrix containing validation (heldout) data
        '''
        return (self.validation_x_batches, self.validation_y_batches)

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_xdata))

        # works only with np.arrays
        # self.train_xdata = self.train_xdata[permutation]
        # self.train_ydata = self.train_ydata[permutation]

        self.train_xdata = [self.train_xdata[i] for i in permutation]
        self.train_ydata = [self.train_ydata[i] for i in permutation]

        self.pointer = 0
