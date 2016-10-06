# This file contains single class with constants used in the project.
class Config():
    # dataset
    data_dir = 'data/bible' # data directory containing input.txt

    # training parameters
    batch_size = 50
    seq_length = 25 # length (in chars) of each text
    embedding_size = 50
    num_epochs = 50
    grad_clip = 5. # clip gradients at this value
    learning_rate = 0.002 # learning rate
    decay_rate = 0.97 # decay rate for rmsprop

    # Model parameters
    model = 'gru'  # rnn, gru, or lstm
    num_layers = 2 # number of layers in the RNN
    rnn_size = 256 # size of RNN hidden state

    # other stuff    
    save_dir = 'save' # directory to store checkpointed models
    save_every = 1000 # save frequency
    init_from = None # continue training from saved model at this path. Path must contain files saved by previous training process:
                        #    'config.pkl'        : configuration;
                        #    'words_vocab.pkl'   : vocabulary definitions;
                        #    'checkpoint'        : paths to model file(s) (created by tf).
                        #                          Note: this file contains absolute paths, be careful when moving files around;
                        #    'model.ckpt-*'      : file(s) with model definition (created by tf)
