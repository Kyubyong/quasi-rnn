# coding: utf-8
from __future__ import print_function
from hyperparams import Hp
import codecs
import re
import numpy as np

def load_vocab():
    # Note that ␀, ␂, ␃, and ⁇  mean padding, EOS, and OOV respectively.
    vocab = u'''␀␃⁇ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÅÇÉÖ×ÜßàáâãäçèéêëíïñóôöøúüýāćČēīœšūβкӒ0123456789!"#$%&''()*+,-./:;=?@[\]^_` ¡£¥©«­®°²³´»¼½¾ยรอ่‒–—‘’‚“”„‟‹›€™♪♫你葱送﻿，'''
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def create_data(source_sents, target_sents): 
    char2idx, idx2char = load_vocab()
    
    X, Y, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        # tokenize naively
        source_sent = re.sub(r"([,.!?])", r" \1", source_sent)
        target_sent = re.sub(r"([,.!?])", r" \1", target_sent)
        
        x = [char2idx.get(char, 2) for char in source_sent + u"␃"] # 2: OOV, ␃: End of text
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
       
def load_train_data():
    de_sents = [line for line in codecs.open(Hp.de_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [line for line in codecs.open(Hp.en_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    X, Y, _, _ = create_data(de_sents, en_sents)
    return X, Y
    
def load_test_data():
    def remove_tags(line):
        line = re.sub("<[^>]+>", "", line) 
        return line.strip()
    
    de_sents = [remove_tags(line) for line in codecs.open(Hp.de_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    en_sents = [remove_tags(line) for line in codecs.open(Hp.en_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]

    X, _, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets # (1064, 150)
     




