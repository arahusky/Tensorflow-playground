# -*- coding: utf-8 -*-

# This class loads, cleans and parses train text into batches.
import os
import collections
from six.moves import cPickle
import numpy as np
import re
import nltk
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

    def build_vocab(self, words):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and all appeared words.
        """

        print(words[0])
        print(type(words[0]))

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

        sentences = nltk.sent_tokenize(data, language='english')

        # Remove too long sentences
        sentences = [sentence for sentence in sentences if len(sentence) < 100]  # removes only about 5 sentences

        # Create labels
        self.labels = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence, language='english')
            self.labels.append(np.array(['1' if word[0].isupper() else '0' for word in words]))

        # convert every letter to lower-case version
        x_text = map((lambda x: x.lower()), sentences)

        sentences_as_seq_of_words = [nltk.word_tokenize(sentence, language='english') for sentence in x_text]
        self.vocab = self.build_vocab(utils.flatten_list_of_lists(sentences_as_seq_of_words))
        words = self.vocab.keys()
        # vocab is a dictionary {"word" : ID}
        # words is just a list of words that appeared in the text
        self.vocab_size = len(words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.vocab, f)

        # inputs is a list of numpy arrays, where each inner array is an ID sequence representing one sentence
        self.inputs = [np.array(map(self.vocab.get, sentence)) for sentence in sentences_as_seq_of_words]

        # BBC articles are sorted according to topic, which makes the classic train/test split of this array topic-biased
        permutation = np.random.permutation(len(self.inputs))
        self.inputs = [self.inputs[i] for i in permutation]
        self.labels = [self.labels[i] for i in permutation]

        print('Saving data')
        # Save data
        with open(data_file, 'wb') as f:
            cPickle.dump((self.inputs, self.labels), f)

    def load_preprocessed(self, vocab_file, data_file):
        with open(vocab_file, 'rb') as f:
            self.vocab = cPickle.load(f)
        self.words = self.vocab.keys()
        self.vocab_size = len(self.words)
        with open(data_file, 'rb') as f:
            self.inputs, self.labels = cPickle.load(f)

    def create_batches(self, train_percentage, valid_percentage, test_percentage):
        # TODO rename me
        num_available_batches = int(len(self.inputs) / self.batch_size)

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
        self.inputs = self.inputs[:max_length]
        self.labels = self.labels[:max_length]

        # split tensor into train, validation and test sets
        self.train_xdata = self.inputs[:num_train_samples]
        self.train_ydata = self.labels[:num_train_samples]

        self.validation_xdata = self.inputs[num_train_samples:num_train_samples + num_validation_samples]
        self.validation_ydata = self.labels[num_train_samples:num_train_samples + num_validation_samples]

        self.test_xdata = self.inputs[num_train_samples + num_validation_samples:]
        self.test_ydata = self.labels[num_train_samples + num_validation_samples:]

        # create numpy array for saving current batch
        print(type(self.train_xdata))
        print(type(self.train_xdata[0]))
        self.max_len = max([row.shape[0] for row in self.train_xdata])  # max len (in # words) of train sentence
        print('self.max_len' + str(self.max_len))
        self.batch_x = np.zeros((self.batch_size, self.max_len), dtype=np.int32)
        self.batch_x_lenghts = np.zeros(self.batch_size, dtype=np.int32)  # contains for each train sentence its length
        self.batch_y = np.zeros((self.batch_size, self.max_len),
                                dtype=np.int32)  # contains for each train sentence its word labels

    def next_batch(self):
        '''
        Returns next train batch.  Each batch consists of three parts:
            batch_x ; contains self.num_batches of train inputs with variable lengths, note that this array has
             dimensions [self.num_batches, max_length], where max_length is a length of the longest sequence in the training set
            batch_x_length: for each sequence in the batch contains its length
            batch_y: contains labels for batch_x inputs
        '''
        for sample_ind in range(self.batch_size):
            self.batch_x_lenghts[sample_ind] = len(self.train_xdata[self.pointer])

            for seq_ind in range(self.batch_x_lenghts[sample_ind]):
                self.batch_x[sample_ind][seq_ind] = self.train_xdata[self.pointer][seq_ind]
                self.batch_y[sample_ind][seq_ind] = self.train_ydata[self.pointer][seq_ind]
            self.pointer += 1
        return self.batch_x, self.batch_x_lenghts, self.batch_y

    def get_test_set(self):
        '''
        Returns list of test batches.
        '''
        testdata_samples = len(self.test_xdata)
        self.test_x_data = np.zeros((testdata_samples, self.max_len))
        self.test_x_data_lengths = np.zeros(len(self.test_xdata))
        self.test_y_data = np.zeros((testdata_samples, self.max_len))

        for sample_ind in range(len(self.test_xdata)):
            self.test_x_data_lengths[sample_ind] = len(self.test_xdata[sample_ind])

            for seq_ind in range(len(self.test_xdata[sample_ind])):
                self.test_x_data[sample_ind][seq_ind] = self.test_xdata[sample_ind][seq_ind]
                self.test_y_data[sample_ind][seq_ind] = self.test_y_data[sample_ind][seq_ind]

        num_testdata_batches = testdata_samples / self.batch_size
        return (
        np.split(self.test_x_data, num_testdata_batches), np.split(self.test_x_data_lengths, num_testdata_batches),
        np.split(self.test_y_data, num_testdata_batches))

    def get_validation_set(self):
        '''
        Returns matrix containing validation (heldout) data
        '''
        return (self.validation_x_batches, self.validation_y_batches)

    def reset_batch_pointer(self):
        # TODO check if sklearn.utils.shuffle is not better
        permutation = np.random.permutation(len(self.train_xdata))

        # works only with np.arrays
        # self.train_xdata = self.train_xdata[permutation]
        # self.train_ydata = self.train_ydata[permutation]

        self.train_xdata = [self.train_xdata[i] for i in permutation]
        self.train_ydata = [self.train_ydata[i] for i in permutation]

        self.pointer = 0
