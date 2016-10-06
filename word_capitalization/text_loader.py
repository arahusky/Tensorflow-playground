# -*- coding: utf-8 -*-

# This class loads, cleans and parses train text into batches.
import os
import collections
from six.moves import cPickle
import numpy as np
import re
import nltk
import config


class TextLoader():
    '''
    This class provides methods to work with the dataset.
    To load dataset, create new instance of this class with provided data_dir pointing to root folder of dataset.
    '''

    # this folder stores twofiles: vocabulary (single words in order as used in vocabulary) and processed tensor file
    CACHE_FOLDER = config.Config.save_dir

    def __init__(self, config, data_dir='bbc'):
        self.data_dir = data_dir
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length

        train_percentage = config.train_
        valid_percentage = config.validation
        test_percentage = config.test

        vocab_file = os.path.join(self.CACHE_FOLDER, 'vocab.pkl')
        data_file = os.path.join(self.CACHE_FOLDER, 'data.npy')

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

    def build_vocab(self, words):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and all appeared words.
        """

        # Build vocabulary
        word_counts = collections.Counter(words)
        # for each word its number of occurence

        # word_counts.most_common() returns tuples in form (word, num_occurence)
        words = [x[0] for x in word_counts.most_common()]
        words = list(sorted(words))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(words)}
        return vocabulary

    def preprocess(self, vocab_file, data_file):
        data = self.load_texts()

        # Text cleaning
        # data = self.clean_str(data)

        x_text = nltk.word_tokenize(data, language='english')
        self.labels = np.array(['1' if word[0].isupper() else '0' for word in x_text])
        # self.labels = np.array(['1' if (word=='the') else '0' for word in x_text])

        # convert all words to lower-case version
        x_text = map((lambda x: x.lower()), x_text)

        self.vocab = self.build_vocab(x_text)
        words = self.vocab.keys()
        # vocab is a dictionary {"word" : ID}
        # words is just a list of words that appeared in the text
        self.vocab_size = len(words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.vocab, f)

        # The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        self.inputs = np.array(list(map(self.vocab.get, x_text)))  # one-dimensional tensor
        # Save the data to data.npy
        np.save(data_file, (self.inputs, self.labels))

    def load_preprocessed(self, vocab_file, data_file):
        with open(vocab_file, 'rb') as f:
            self.vocab = cPickle.load(f)
        self.words = self.vocab.keys()
        self.vocab_size = len(self.words)
        self.inputs, self.labels = np.load(data_file)

    def create_batches(self, train_percentage, valid_percentage, test_percentage):
        num_available_batches = int(self.inputs.size / (self.batch_size *
                                                        self.seq_length))

        # counts in means of batches
        self.num_batches = int(round(num_available_batches * train_percentage))  # number of train batches prepared
        num_validation_batches = int(
            round(num_available_batches * valid_percentage))  # number of validation batches prepared
        num_test_batches = num_available_batches - self.num_batches - num_validation_batches  # number of test batches prepared

        # counts in means of samples
        num_train_samples = self.num_batches * self.batch_size * self.seq_length
        num_validation_samples = num_validation_batches * self.batch_size * self.seq_length
        num_test_samples = num_test_batches * self.batch_size * self.seq_length

        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size smaller."

        print(str(self.num_batches) + ' train batches available')
        print(str(num_validation_batches) + ' validation batches available')
        print(str(num_test_batches) + ' test batches available')
        # cut end part after the last expected batch to ensure that the data can be splitted into batches with no padding
        max_length = num_available_batches * self.batch_size * self.seq_length  # how many samples may be prepared
        self.inputs = self.inputs[:max_length]
        self.labels = self.labels[:max_length]

        # cut tensor in train, validation and test sets
        train_xdata = self.inputs[:num_train_samples]
        train_ydata = self.labels[:num_train_samples]

        validation_xdata = self.inputs[num_train_samples:num_train_samples + num_validation_samples]
        validation_ydata = self.labels[num_train_samples:num_train_samples + num_validation_samples]

        test_xdata = self.inputs[num_train_samples + num_validation_samples:]
        test_ydata = self.labels[num_train_samples + num_validation_samples:]
        print(train_xdata.shape, train_ydata.shape, validation_xdata.shape, validation_ydata.shape, test_xdata.shape,
              test_ydata.shape)

        # to create batches, first reshape data to have self.batch_size rows (each column contains one input),
        # and then split this large matrix with split into self.num_batches (split along columns)
        self.train_x_batches = np.split(train_xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.train_y_batches = np.split(train_ydata.reshape(self.batch_size, -1), self.num_batches, 1)

        if num_validation_batches != 0:  # no validation data required
            self.validation_x_batches = validation_xdata.reshape(num_validation_batches * self.batch_size, -1)
            self.validation_y_batches = validation_ydata.reshape(num_validation_batches * self.batch_size, -1)
        else:
            self.validation_x_batches = np.array([])
            self.validation_y_batches = np.array([])

        self.test_x_batches = test_xdata.reshape(num_test_batches * self.batch_size, -1)
        self.test_y_batches = test_ydata.reshape(num_test_batches * self.batch_size, -1)

    def next_batch(self):
        '''
        Returns next train batch.
        '''
        x, y = self.train_x_batches[self.pointer], self.train_y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def get_test_set(self):
        '''
        Returns matrix containing test data.
        '''
        return (self.test_x_batches, self.test_y_batches)

    def get_validation_set(self):
        '''
        Returns matrix containing validation (heldout) data
        '''
        return (self.validation_x_batches, self.validation_y_batches)

    def reset_batch_pointer(self):
        self.pointer = 0
        # todo possibly permutate inside array
