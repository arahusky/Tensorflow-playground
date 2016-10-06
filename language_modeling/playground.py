# This file shows basic usage of this project

# The first thing one must do, is train a model
# Two types of models are supported: char and word level 
# Training with default parameters (which can be changed in config file) lasts about half an hour. To skip this, one can use already trained models (defaulty saved in save folder), thus no training is required. Once model is trained, it is saved in 'save' directory.

import train, os
# train(is_char_model = True)

# Load trained model
from model import Model
from six.moves import cPickle

with open(os.path.join('save', 'config.pkl'), 'rb') as f:
    saved_args = cPickle.load(f)
        
model = Model(saved_args, True)

# The model can then be used for two purposes:
import utils
from config import Config
config = Config()

# 1. score given sequence
# meaningful one should have higher probability
print('Probabibility of sequence "Is nice" is: ', utils.compute_seq_prob(['I', 's', ' ', 'n', 'i', 'c', 'e'], model, config))
print('Probabibility of sequence "Is pnce" is: ', utils.compute_seq_prob(['I', 's', ' ', 'p', 'n', 'c', 'e'], model, config))

# 2. sample text

config.n = 2000 # number of characters (words in case of word-level LM) to sample
config.prime = 'A' # starting character
config.sample = 1 # way, how to choose probable text: 0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces

utils.sample(config, model, is_char_model = True)
