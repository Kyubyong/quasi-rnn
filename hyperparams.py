class Hyperparams:
    '''Hyperparameters'''
    de_train = 'corpora/train.tags.de-en.de'
    en_train = 'corpora/train.tags.de-en.en'
    de_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    en_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    maxlen = 150 # Maximum sentence length
    batch_size = 32
    hidden_units = 320
    num_blocks = 7
    num_epochs = 10
    lr = 0.0001
    logdir = 'log'
    savedir = 'ckpt'
