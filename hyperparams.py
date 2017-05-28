
class Hp:
    '''Hyperparameters'''
    de_train = 'corpora/train.tags.de-en.de'
    en_train = 'corpora/train.tags.de-en.en'
    de_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    en_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    maxlen = 150 # Maximum sentence length
    batch_size = 16
    hidden_units = 320
