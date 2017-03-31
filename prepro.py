# coding: utf-8
from __future__ import print_function
from hyperparams import Hp
import codecs
import re
import numpy as np

def load_vocab():
    # Note that ␀, ␂, ␃, and ⁇  mean padding, BOS, EOS, and OOV respectively.
    vocab = u'''␀␃⁇ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÅÇÉÖ×ÜßàáâãäçèéêëíïñóôöøúüýāćČēīœšūβкӒ0123456789!"#$%&''()*+,-./:;=?@[\]^_` ¡£¥©«­®°²³´»¼½¾ยรอ่‒–—‘’‚“”„‟‹›€™♪♫你葱送﻿，'''
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def create_data(source_sents, target_sents, input_reverse=False): 
    char2idx, idx2char = load_vocab()
    
    X, Y, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [char2idx.get(char, 2) for char in source_sent + u"␃"] # 2: OOV, ␃: End of text
        if input_reverse:
            x = x[::-1][1:] + x[-1:]
        y = [char2idx.get(char, 2) for char in target_sent + u"␃"] 
        if max(len(x), len(y)) <= Hp.maxlen:
            x += [0] * (Hp.maxlen - len(x)) # zero postpadding
            y += [0] * (Hp.maxlen - len(y)) 
            
            X.append(x); Y.append(y)
            Sources.append(source_sent)
            Targets.append(target_sent)
    X = np.array(X, np.int32)
    Y = np.array(Y, np.int32)
    
    print("X.shape =", X.shape) 
    print("Y.shape =", Y.shape) 
    
    return X, Y, Sources, Targets
       
def load_train_data(input_reverse=False):
    de_sents = [line for line in codecs.open(Hp.de_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [line for line in codecs.open(Hp.en_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    X, Y, _, _ = create_data(de_sents, en_sents, input_reverse=input_reverse)
    return X, Y
    
def load_test_data(input_reverse=False):
    def remove_tags(line):
        line = re.sub("<[^>]+>", "", line) 
        return line.strip()
    
    de_sents = [remove_tags(line) for line in codecs.open(Hp.de_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    en_sents = [remove_tags(line) for line in codecs.open(Hp.en_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]

    X, _, Sources, Targets = create_data(de_sents, en_sents, input_reverse=input_reverse)
    return X, Sources, Targets # (1064, 150)
     




