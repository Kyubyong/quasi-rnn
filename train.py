# -*- coding: utf-8 -*-
from __future__ import print_function
import sugartensor as tf
import numpy as np
from prepro import *

# set log level to debug
tf.sg_verbosity(10)

def get_batch_data():
    '''
    Returns:
      A Tuple of X batch queues (Tensor), Y batch queues (Tensor), 
      and number of batches (int) 
    '''
    # Load data
    X, Y = load_train_data()
    char2idx, idx2char = load_vocab()
    
    # Make slice 
    x_q, y_q = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int32),
                                          tf.convert_to_tensor(Y, tf.int32)])

    # Create batch queues
    x, y = tf.train.shuffle_batch([x_q, y_q],
                                num_threads=8,
                                batch_size=Hp.bs, 
                                capacity=Hp.bs*64,
                                min_after_dequeue=Hp.bs*32, 
                                allow_smaller_final_batch=False)
    
    # Get number of mini-batches
    num_batch = len(X) // Hp.bs
    
    return x, y, num_batch

@tf.sg_layer_func
def sg_quasi_conv1d(tensor, opt):
    opt += tf.sg_opt(is_enc=False)
    # Split into H and H_zfo
    H = tensor[:Hp.bs]
    H_z = tensor[Hp.bs:2*Hp.bs]
    H_f = tensor[2*Hp.bs:3*Hp.bs]
    H_o = tensor[3*Hp.bs:]
    if opt.is_enc:
        H_z, H_f, H_o = 0, 0, 0
    
    # Convolution and merging
    with tf.sg_context(act="linear", causal=(not opt.is_enc), bn=opt.is_enc, ln=(not opt.is_enc)):
        Z = H.sg_aconv1d() + H_z # (16, 300, 320)
        F = H.sg_aconv1d() + H_f # (16, 300, 320)
        O = H.sg_aconv1d() + H_o # (16, 300, 320)

    # Activation
    Z = Z.sg_bypass(act="tanh") # (16, 300, 320)
    F = F.sg_bypass(act="sigmoid") # (16, 300, 320)
    O = O.sg_bypass(act="sigmoid") # (16, 300, 320)
    
    # Masking
    M = tf.sign(tf.abs(H))[:, :, :1] # (16, 300, 1) float32. 0 or 1
    Z *= M # broadcasting
    F *= M # broadcasting
    O *= M # broadcasting
    
    # Concat
    ZFO = tf.concat(axis=0, values=[Z, F, O])
    
    return ZFO # (16*3, 150, 320)

# injection
tf.sg_inject_func(sg_quasi_conv1d)
    
@tf.sg_rnn_layer_func
def sg_quasi_rnn(tensor, opt):
    # Split
    if opt.att:
        H, Z, F, O = tf.split(axis=0, num_or_size_splits=4, value=tensor) # (16, 150, 320) for all
    else:
        Z, F, O = tf.split(axis=0, num_or_size_splits=3, value=tensor) # (16, 150, 320) for all
    
    # step func
    def step(z, f, o, c):
        '''
        Runs fo-pooling at each time step
        '''
        c = f * c + (1 - f) * z
        
        if opt.att: # attention
            a = tf.nn.softmax(tf.einsum("ijk,ik->ij", H, c)) # alpha. (16, 150) 
            k = (a.sg_expand_dims() * H).sg_sum(dims=1) # attentional sum. (16, 150) 
            h = o * (k.sg_dense(act="linear") + c.sg_dense(act="linear"))
        else:
            h = o * c
        
        return h, c # hidden states, (new) cell memories
    
    # Do rnn loop
    c, hs = 0, []
    timesteps = tensor.get_shape().as_list()[1]
    for t in range(timesteps):
        z = Z[:, t, :] # (16, 320)
        f = F[:, t, :] # (16, 320)
        o = O[:, t, :] # (16, 320)

        # apply step function
        h, c = step(z, f, o, c) # (16, 320), (16, 320)
        
        # save result
        hs.append(h.sg_expand_dims(dim=1))
    
    # Concat to return    
    H = tf.concat(axis=1, values=hs) # (16, 150, 320)
    if opt.is_enc:
        H_z = tf.tile((h.sg_dense(act="linear").sg_expand_dims(dim=1)), [1, timesteps, 1])
        H_f = tf.tile((h.sg_dense(act="linear").sg_expand_dims(dim=1)), [1, timesteps, 1])
        H_o = tf.tile((h.sg_dense(act="linear").sg_expand_dims(dim=1)), [1, timesteps, 1])
        
        concatenated = tf.concat(axis=0, values=[H, H_z, H_f, H_o]) # (16*4, 150, 320)
        return concatenated
    else:
        return H # (16, 150, 320)
    
# injection
tf.sg_inject_func(sg_quasi_rnn)

class Graph(object):
    def __init__(self, mode="train"):
        # Inputs and Labels
        if mode == "train":
            self.x, self.y, self.num_batch = get_batch_data() # (16, 150) int32, (16, 150) int32, int
            self.y_src = tf.concat(axis=1, values=[tf.zeros((Hp.bs, 1), tf.int32), self.y[:, :-1]]) # (16, 150) int32
        else: # inference
            self.x = tf.placeholder(tf.int32, shape=(Hp.bs, Hp.maxlen))
            self.y_src = tf.placeholder(tf.int32, shape=(Hp.bs, Hp.maxlen))
        
        # Load vocabulary    
        self.char2idx, self.idx2char = load_vocab()
        
        # Embedding
        self.emb_x = tf.sg_emb(name='emb_x', voca_size=len(self.char2idx), dim=Hp.hd)  # (179, 320)
        self.emb_y = tf.sg_emb(name='emb_y', voca_size=len(self.char2idx), dim=Hp.hd)  # (179, 320)
        self.X = self.x.sg_lookup(emb=self.emb_x) # (16, 150, 320)
        self.Y = self.y_src.sg_lookup(emb=self.emb_y) # (16, 150, 320)
            
        # Encoding
        self.conv = self.X.sg_quasi_conv1d(is_enc=True, size=6) # (16*4, 150, 320)
        self.pool = self.conv.sg_quasi_rnn(is_enc=True, att=False) # (16*4, 150, 320)
        self.H_zfo1 = self.pool[Hp.bs:] # (16*3, 15, 320) for decoding
         
        self.conv = self.pool.sg_quasi_conv1d(is_enc=True, size=2) # (16*4, 150, 320)
        self.pool = self.conv.sg_quasi_rnn(is_enc=True, att=False) # (16*4, 150, 320)
        self.H_zfo2 = self.pool[Hp.bs:] # (16*3, 150, 320) for decoding
         
        self.conv = self.pool.sg_quasi_conv1d(is_enc=True, size=2) # (16*4, 150, 320)
        self.pool = self.conv.sg_quasi_rnn(is_enc=True, att=False) # (16*4, 150, 320)
        self.H_zfo3 = self.pool[Hp.bs:] # (16*3, 150, 320) for decoding
         
        self.conv = self.pool.sg_quasi_conv1d(is_enc=True, size=2) # (16*4, 150, 320)
        self.pool = self.conv.sg_quasi_rnn(is_enc=True, att=False) # (16*4, 150, 320)
        self.H4 = self.pool[:Hp.bs]
        self.H_zfo4 = self.pool[Hp.bs:] # (16*3, 150, 320) for decoding

        # Decoding
        self.dec = self.Y.sg_concat(target=self.H_zfo1, dim=0)
                     
        self.d_conv = self.dec.sg_quasi_conv1d(is_enc=False, size=2)
        self.d_pool = self.d_conv.sg_quasi_rnn(is_enc=False, att=False) # (16*4, 150, 320)
        
        self.d_conv = (self.d_pool.sg_concat(target=self.H_zfo2, dim=0)
                                  .sg_quasi_conv1d(is_enc=False, size=2))
        self.d_pool = self.d_conv.sg_quasi_rnn(is_enc=False, att=False) # (16*4, 150, 320)
        
        self.d_conv = (self.d_pool.sg_concat(target=self.H_zfo3, dim=0)
                                  .sg_quasi_conv1d(is_enc=False, size=2))
        self.d_pool = self.d_conv.sg_quasi_rnn(is_enc=False, att=False) # (16*4, 150, 320)
        
        self.d_conv = (self.d_pool.sg_concat(target=self.H_zfo4, dim=0)
                                  .sg_quasi_conv1d(is_enc=False, size=2))
        self.concat = self.H4.sg_concat(target=self.d_conv, dim=0)
        self.d_pool = self.concat.sg_quasi_rnn(is_enc=False, att=True) # (16*4, 150, 320)
        
        self.logits = self.d_pool.sg_conv1d(size=1, dim=len(self.char2idx), act="linear") # (16, 150, 179)
        self.preds = self.logits.sg_argmax()
        if mode=='train':
            # cross entropy loss with logits ( for training set )
            self.loss = self.logits.sg_ce(target=self.y, mask=True)
            self.istarget = tf.not_equal(self.y, 0).sg_float()
            self.reduced_loss = (self.loss.sg_sum()) / (self.istarget.sg_sum() + 0.00001)
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")

def main():
    g = Graph(); print("Graph Loaded")
    tf.sg_train(optim="Adam", lr=0.0001, lr_reset=True, loss=g.reduced_loss, ep_size=g.num_batch,
                save_dir='asset/train', max_ep=10, early_stop=False)
    
if __name__ == "__main__":
    main(); print("Done")
