# -*- coding: utf-8 -*-
from __future__ import print_function
from hyperparams import Hyperparams as hp
import codecs
import re
import pickle
import numpy as np

def load_vocab():
    # Note that ␀, ␂, ␃, and ⁇  mean padding, SOS, EOS, and OOV respectively.
    vocab = u'''␀␂␃⁇ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÅÇÉÖ×ÜßàáâãäçèéêëíïñóôöøúüýāćČēīœšūβкӒ0123456789!"#$%&''()*+,-./:;=?@[\]^_` ¡£¥©«­®°²³´»¼½¾ยรอ่‒–—‘’‚“”„‟‹›€™♪♫你葱送﻿，'''
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def create_data(source_sents, target_sents): 
    char2idx, idx2char = load_vocab()
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [char2idx.get(char, 3) for char in source_sent + u"␃"] # 3: OOV, ␃: End of Text
        y = [char2idx.get(char, 3) for char in target_sent + u"␃"] 
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    print("X.shape =", X.shape) 
    print("Y.shape =", Y.shape) 
    
    return X, Y, Sources, Targets

def create_eval_data(source_sents, target_sents): 
    char2idx, idx2char = load_vocab()
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [char2idx.get(char, 3) for char in source_sent + u"␃"] # 3: OOV, ␃: End of Text
        y = [char2idx.get(char, 3) for char in target_sent + u"␃"] 
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    print("X.shape =", X.shape) 
    print("Y.shape =", Y.shape) 
    
    return X, Y, Sources, Targets
       
def load_train_data():
    de_sents = [line for line in codecs.open(hp.de_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [line for line in codecs.open(hp.en_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y
    
def load_test_data():
    def _remove_tags(line):
        line = re.sub("<[^>]+>", "", line) 
        return line.strip()
    
    de_sents = [_remove_tags(line) for line in codecs.open(hp.de_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    en_sents = [_remove_tags(line) for line in codecs.open(hp.en_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets # (1064, 150)

     




