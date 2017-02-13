# coding: utf-8
from __future__ import print_function
import codecs
import re
import pickle
import numpy as np

class Hp:
    """Hyperparameters"""
    bs = 16 # batch size
    hd = 320 # hidden dimension
    maxlen = 150 # Maximum sentence length
    de_train = 'corpora/train.tags.de-en.de'
    en_train = 'corpora/train.tags.de-en.en'
    de_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    en_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'

def load_vocab():
    # Note that ␀, ␂, ␃, ⁇  mean padding, BOS, EOS, and OOV respectively.
    vocab = u'''␀␂␃⁇ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÅÇÉÖ×ÜßàáâãäçèéêëíïñóôöøúüýāćČēīœšūβкӒ0123456789!"#$%&''()*+,-./:;=?@[\]^_` ¡£¥©«­®°²³´»¼½¾ยรอ่‒–—‘’‚“”„‟‹›€™♪♫你葱送﻿，'''
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def create_data(source_sents, target_sents): 
    char2idx, idx2char = load_vocab()
    
    X, Y, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [char2idx.get(char, 3) for char in u"␂" + source_sent] # 3: OOV
        y = [char2idx.get(char, 3) for char in target_sent + u"␃"] # 3: OOV
        if max(len(x), len(y)) <= Hp.maxlen:
            x = [0] * (Hp.maxlen - len(x)) + x # zero prepadding
            y += [0] * (Hp.maxlen - len(y)) # zero postpadding
            
            X.append(x)
            Y.append(y)
            Sources.append(source_sent)
            Targets.append(target_sent)
            
    X = np.array(X, np.int32)
    Y = np.array(Y, np.int32)
    
    print("X.shape =", X.shape) # (157014, 150)
    print("Y.shape =", Y.shape) # (157014, 150)
    return X, Y, Sources, Targets
       
def create_train_data():
    de_sents = [line for line in codecs.open(Hp.de_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [line for line in codecs.open(Hp.en_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    
    X, Y, _, _ = create_data(de_sents, en_sents)
    np.savez('data/train.npz', X=X, Y=Y)
    
def load_train_data():
    X = np.load('data/train.npz')['X']
    Y = np.load('data/train.npz')['Y']
    return X, Y

def create_test_data():
    def remove_tags(line):
        line = re.sub("<[^>]+>", "", line) 
        return line.strip()
    
    de_sents = [remove_tags(line) for line in codecs.open(Hp.de_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    en_sents = [remove_tags(line) for line in codecs.open(Hp.en_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]

    X, _, Sources, Targets = create_data(de_sents, en_sents)
    pickle.dump((X, Sources, Targets), open('data/test.pkl', 'wb'))
  
def load_test_data():
    X, Sources, Targets = pickle.load(open('data/test.pkl', 'rb'))
    return X, Sources, Targets

if __name__ == "__main__":
    create_train_data()
    create_test_data()
    print("Done!")
     




