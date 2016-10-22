# This file contains single class with constants used in the project.
class Config():
    # train / valid / test
    train_= 0.75
    validation = 0.0
    test = 0.25

    # training parameters
    batch_size = 50
    embedding_size = 100
    num_epochs = 20
    keep_prob = 0.5 # dropout keep probability during training
    grad_clip = 5. # clip gradients at this value
    learning_rate = 1e-4  # learning rate

    # Model parameters
    model = 'gru'  # rnn, gru, or lstm
    num_layers = 3 # number of layers in the RNN
    rnn_size = 200 # size of RNN hidden state

    # dataset
    max_words_in_sentence = 60 # remove all sentences having more words than this constant

    # other stuff
    log_dir = 'logs' # to store TensorBoard logs
    save_dir = 'save' # directory to store checkpointed models
    save_every = 1000 # save frequency
    init_from = None # continue training from saved model at this path. Path must contain files saved by previous training process:
                        #    'config.pkl'        : configuration;
                        #    'words_vocab.pkl'   : vocabulary definitions;
                        #    'checkpoint'        : paths to model file(s) (created by tf).
                        #                          Note: this file contains absolute paths, be careful when moving files around;
#    'model.ckpt-*'      : file(s) with model definition (created by tf)

    # TODO implement for logs
    def to_string(self):
        return ""
