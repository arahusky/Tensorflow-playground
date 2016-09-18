# -*- coding: utf-8 -*-

# This class loads, cleans and parses train text into batches.
import os
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools

class TextLoader():
    '''
    This class provides methods to work with the dataset. 
    This class can either interpret text as a sequence of words, or a sequence of chars. When working with chars, initialize this class with is_char_model set to True.

    To load dataset, create new instance of this class with provided data_dir.  
    To get next batch, call next_batch method.
    '''
    def __init__(self, data_dir, batch_size, seq_length, is_char_model = False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file, is_char_model)
        else:
            print("loading preprocessed files") 
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
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
        return string.strip().lower()


    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and all appeared words.
        """
        # Build vocabulary
        word_counts = collections.Counter(sentences)
        # for each word its number of occurence
        
        # word_counts.most_common() returns tuples in form (word, num_occurence)
        words = [x[0] for x in word_counts.most_common()]
        words = list(sorted(words))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(words)}
        return [vocabulary, words]

    def preprocess(self, input_file, vocab_file, tensor_file, is_char_model):
        with open(input_file, "r") as f:
            data = f.read()

        # Optional text cleaning or make them lower case, etc.
        # data = self.clean_str(data)
        x_text = data
        if not is_char_model:
            x_text = x_text.split()

        self.vocab, self.words = self.build_vocab(x_text)
        # vocab is a dictionary {"word" : ID}
        # words is just a list of words that appeared in the text
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        self.tensor = np.array(list(map(self.vocab.get, x_text))) # one-dimensional tensor
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # cut end part after the last expected batch to ensure that the data can be splitted into batches with no padding
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
    
        # xdata, ydata are now same 1-dimensional arrays storing text as sequence of integers
        # we want ydata to contain xdata shifted right by 1, so that y[i] says what comes after x[i]
        # e.g. for xdata = 012345, ydata = 123450
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
                
        # to create batches, first reshape data to have self.batch_size rows (each column contains one input),
        # and then split this large matrix with split into self.num_batches (split along columns)
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
