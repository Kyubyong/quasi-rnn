# -*- coding: utf-8 -*-
from __future__ import print_function
from hyperparams import Hp
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
    
    # Get number of mini-batches
    num_batch = len(X) // Hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Make slice 
    x, y = tf.train.slice_input_producer([X, Y])

    # Create batch queues
    x, y = tf.train.shuffle_batch([x, y],
                                num_threads=8,
                                batch_size=Hp.batch_size, 
                                capacity=Hp.batch_size*64,
                                min_after_dequeue=Hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    return x, y, num_batch

@tf.sg_layer_func
def sg_quasi_conv1d(tensor, opt):
    '''
    Args:
      tensor: A 3-D tensor of either [batch size, time steps, embedding size] for original
          X or [batch size * 4, time steps, embedding size] for the others.
           
    '''
    opt += tf.sg_opt(is_enc=False)
    
    # Split into H and H_zfo
    H = tensor[:Hp.batch_size]
    H_z = tensor[Hp.batch_size:2*Hp.batch_size]
    H_f = tensor[2*Hp.batch_size:3*Hp.batch_size]
    H_o = tensor[3*Hp.batch_size:]
    if opt.is_enc:
        H_z, H_f, H_o = 0, 0, 0
    
    # Convolution and merging
    with tf.sg_context(size=opt.size, act="linear", causal=(not opt.is_enc)):
        Z = H.sg_aconv1d() + H_z # (16, 150, 320)
        F = H.sg_aconv1d() + H_f # (16, 150, 320)
        O = H.sg_aconv1d() + H_o # (16, 150, 320)

    # Activation
    Z = Z.sg_bypass(act="tanh") # (16, 150, 320)
    F = F.sg_bypass(act="sigmoid") # (16, 150, 320)
    O = O.sg_bypass(act="sigmoid") # (16, 150, 320)
    
    # Masking
    #M = tf.sign(tf.abs(tf.reduce_sum(H, axis=-1, keep_dims=True))) # (16, 150, 1) float32. 0 or 1
    #Z *= M # broadcasting
    #F *= M # broadcasting
    #O *= M # broadcasting
    
    # Concat
    ZFO = tf.concat([Z, F, O], 0)
    
    return ZFO # (16*3, 150, 320)

# injection
tf.sg_inject_func(sg_quasi_conv1d)
    
@tf.sg_rnn_layer_func
def sg_quasi_rnn(tensor, opt):
    # Split
    if opt.att:
        H, Z, F, O = tf.split(tensor, 4, axis=0) # (16, 150, 320) for all
    else:
        Z, F, O = tf.split(tensor, 3, axis=0) # (16, 150, 320) for all
    
#     M = tf.sign(tf.abs(tf.reduce_sum(Z, axis=-1, keep_dims=True))) 
    # step func
    def step(z, f, o, c):
        '''
        Runs fo-pooling at each time step
        '''
        c = f * c + (1 - f) * z
        
        if opt.att: # attention
            a = tf.nn.softmax(tf.einsum("ijk,ik->ij", H, c)) # alpha. (16, 150) 
            k = (a.sg_expand_dims() * H).sg_sum(axis=1) # attentional sum. (16, 320) 
            h = o * (k.sg_dense(act="linear") + \
                     c.sg_dense(act="linear"))
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
        hs.append(h.sg_expand_dims(axis=1))
    
    # Concat to return    
    H = tf.concat(hs, 1) # (16, 150, 320)
    #seqlen = tf.to_int32(tf.reduce_sum(tf.sign(tf.abs(tf.reduce_sum(H, axis=-1))), 1)) # (16,) float32
    #h = tf.reverse_sequence(input=H, seq_length=seqlen, seq_dim=1)[:, 0, :] # last hidden state vector
    
    if opt.is_enc: 
        H_z = tf.tile((h.sg_dense(act="linear").sg_expand_dims(axis=1)), [1, timesteps, 1])
        H_f = tf.tile((h.sg_dense(act="linear").sg_expand_dims(axis=1)), [1, timesteps, 1])
        H_o = tf.tile((h.sg_dense(act="linear").sg_expand_dims(axis=1)), [1, timesteps, 1])
        concatenated = tf.concat([H, H_z, H_f, H_o], 0) # (16*4, 150, 320)
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
            self.y_src = tf.concat([tf.zeros((Hp.batch_size, 1), tf.int32), self.y[:, :-1]], 1) # (16, 150) int32
        else: # inference
            self.x = tf.placeholder(tf.int32, shape=(Hp.batch_size, Hp.maxlen))
            self.y_src = tf.placeholder(tf.int32, shape=(Hp.batch_size, Hp.maxlen))
        
        # Load vocabulary    
        char2idx, idx2char = load_vocab()
        
        # Embedding
        def embed(inputs, vocab_size, embed_size, variable_scope):
            '''
            inputs = tf.expand_dims(tf.range(5), 0) => (1, 5)
            _embed(inputs, 5, 10) => (1, 5, 10)
            '''
            with tf.variable_scope(variable_scope):
                lookup_table = tf.get_variable('lookup_table', 
                                               dtype=tf.float32, 
                                               shape=[vocab_size, embed_size],
                                               initializer=tf.truncated_normal_initializer())
            return tf.nn.embedding_lookup(lookup_table, inputs)
        
        X = embed(self.x, vocab_size=len(char2idx), embed_size=Hp.hidden_units, variable_scope='X')  # (179, 320)
        Y = embed(self.y_src, vocab_size=len(char2idx), embed_size=Hp.hidden_units, variable_scope='Y')  # (179, 320)
#         Y = tf.concat((tf.zeros_like(Y[:, :1, :]), Y[:, :-1, :]), 1)
            
        # Encoding
        conv = X.sg_quasi_conv1d(is_enc=True, size=6) # (16*3, 150, 320)
        pool = conv.sg_quasi_rnn(is_enc=True, att=False) # (16*4, 150, 320)
        H_zfo1 = pool[Hp.batch_size:] # (16*3, 15, 320) for decoding
         
        conv = pool.sg_quasi_conv1d(is_enc=True, size=2) # (16*3, 150, 320)
        pool = conv.sg_quasi_rnn(is_enc=True, att=False) # (16*4, 150, 320)
        H_zfo2 = pool[Hp.batch_size:] # (16*3, 150, 320) for decoding
         
        conv = pool.sg_quasi_conv1d(is_enc=True, size=2) # (16*3, 150, 320)
        pool = conv.sg_quasi_rnn(is_enc=True, att=False) # (16*4, 150, 320)
        H_zfo3 = pool[Hp.batch_size:] # (16*3, 150, 320) for decoding
         
        conv = pool.sg_quasi_conv1d(is_enc=True, size=2) # (16*3, 150, 320)
        pool = conv.sg_quasi_rnn(is_enc=True, att=False) # (16*4, 150, 320)
        H4 = pool[:Hp.batch_size] # (16, 150, 320) for decoding
        H_zfo4 = pool[Hp.batch_size:] # (16*3, 150, 320) for decoding

        # Decoding
        d_conv = (Y.sg_concat(target=H_zfo1, axis=0)
                   .sg_quasi_conv1d(is_enc=False, size=2))
        d_pool = d_conv.sg_quasi_rnn(is_enc=False, att=False) # (16*4, 150, 320)
        
        d_conv = (d_pool.sg_concat(target=H_zfo2, axis=0)
                        .sg_quasi_conv1d(is_enc=False, size=2))
        d_pool = d_conv.sg_quasi_rnn(is_enc=False, att=False) # (16*4, 150, 320)
        
        d_conv = (d_pool.sg_concat(target=H_zfo3, axis=0)
                        .sg_quasi_conv1d(is_enc=False, size=2))
        d_pool = d_conv.sg_quasi_rnn(is_enc=False, att=False) # (16*4, 150, 320)
        
        d_conv = (d_pool.sg_concat(target=H_zfo4, axis=0)
                        .sg_quasi_conv1d(is_enc=False, size=2))
        concat = H4.sg_concat(target=d_conv, axis=0)
        d_pool = concat.sg_quasi_rnn(is_enc=False, att=True) # (16, 150, 320)
        
        logits = d_pool.sg_conv1d(size=1, dim=len(char2idx), act="linear") # (16, 150, 179)

        if mode=='train':
            # cross entropy loss with logits ( for training set )
            self.loss = logits.sg_ce(target=self.y, mask=True)
            istarget = tf.not_equal(self.y, 0).sg_float()
            self.reduced_loss = (self.loss.sg_sum()) / (istarget.sg_sum() + 1e-8)
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")
        else: # inference
            self.preds = logits.sg_argmax() 

def main():
    g = Graph(); print("Graph Loaded")
    tf.sg_train(optim="Adam", lr=0.0001, lr_reset=True, loss=g.reduced_loss, ep_size=g.num_batch,
                save_dir='asset/train', max_ep=10, early_stop=False)
    
if __name__ == "__main__":
    main(); print("Done")
