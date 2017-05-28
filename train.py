# -*- coding: utf-8 -*-
'''
This is a TensorFlow implementation of 
Character-Level Machine Translation in the paper 
'Neural Machine Translation in Linear Time' (version updated in 2017)
https://arxiv.org/abs/1610.10099. 

Note that I've changed a line in the file.
`tensorflow/contrib/layers/python/layers/layer.py` for some reason.
Check below.

line 1532
Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)

By kyubyong park. kbpark.linguist@gmail.com. https://www.github.com/kyubyong/bytenet
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
from prepro import *
import os
from tqdm import tqdm

def get_batch_data():
    # Load data
    X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (64, 100), (64, 100), ()

def embed(inputs, vocab_size, embed_size, scope="embed"):
    '''
    Args:
      tensor: A 2-D tensor of [batch, time].
      vocab_size: An int. The number of vocabulary.
      num_units: An int. The number of embedding units.
 
    Returns:
      An embedded tensor whose index zero is associated with constant 0. 
    '''
    with tf.variable_scope(scope):
        lookup_table_for_zero = tf.zeros(shape=[1, embed_size], dtype=tf.float32)
        lookup_table_for_others = tf.get_variable('lookup_table', 
                                            dtype=tf.float32, 
                                            shape=[vocab_size-1, embed_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
        lookup_table = tf.concat((lookup_table_for_zero, lookup_table_for_others), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)
    
def normalize_activate(inputs, scope="norm1"):
    '''
    Args:
      tensor: A 3-D or 4-D tensor.
    
    Returns:
      A tensor of the same shape as `tensor`, which has been 
      layer normalized and subsequently activated by Relu.
    '''
    return tf.contrib.layers.layer_norm(inputs=inputs, center=True, scale=True, 
                                        activation_fn=tf.nn.relu, scope=scope)

def conv1d(inputs, 
           filters, 
           size=1, 
           rate=1, 
           padding="SAME", 
           causal=False,
           use_bias=False,
           scope="conv1d"):
    '''
    Args:
      inputs: A 3-D tensor of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `SAME` or `VALID`.
      causal: A boolean. If True, zeros of (kernel size - 1) * rate are padded on the left
        for causality.
      use_bias: A boolean.
    
    Returns:
      A masked tensor of the sampe shape as `tensor`.
    '''
    
    with tf.variable_scope(scope):
        if causal:
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "VALID"
            
        params = {"inputs":inputs, "filters":filters, "kernel_size":size,
                "dilation_rate":rate, "padding":padding, "activation":None, 
                "use_bias":use_bias}
        
        out = tf.layers.conv1d(**params)
    
    return out

def block(tensor, 
          size=3, 
          rate=1, 
          initial=False, 
          causal=False,
          scope="block1"):
    '''
    Refer to Figure 3 on page 4 of the original paper.
    Args
      tensor: A 3-D tensor of [batch, time, depth].
      size: An int. Filter size.
      rate: An int. Dilation rate.
      initial: A boolean. If True, `tensor` will not be activated at first.
      is_training: A boolean. Phase declaration for batch normalization.
      normalization_type: Either `ln` or `bn`.
      causal: A boolean. If True, zeros of (kernel size - 1) * rate are prepadded
        for causality.
    
    Returns
      A tensor of the same shape as `tensor`.
    '''
    with tf.variable_scope(scope):
        out = tensor
        
        # input dimension
        in_dim = out.get_shape().as_list()[-1]
        
        if not initial:
            out = normalize_activate(out, scope="norm_1")
        
        # 1 X 1 convolution -> Dimensionality reduction
        out = conv1d(out, filters=in_dim/2, size=1, causal=causal, scope="conv1d_1")
        
        # normalize and activate
        out = normalize_activate(out, scope="norm_2")
        
        # 1 X k convolution
        out = conv1d(out, filters=in_dim/2, size=size, rate=rate, causal=causal, scope="conv1d_2")
        
        # normalize and activate
        out = normalize_activate(out, scope="norm_3")
        
        # 1 X 1 convolution -> Dimension recovery
        out = conv1d(out, filters=in_dim, size=1, causal=causal, scope="conv1d_3")
        
        # Residual connection
        out += tensor
    
    return out 

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data() # (N, T)
                self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1) # 2: BOS
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.decoder_inputs = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            
            # Load vocabulary    
            char2idx, idx2char = load_vocab()
             
            # Embedding
            self.enc = embed(self.x, len(char2idx), hp.hidden_units, scope="embed_enc")
            self.dec = embed(self.decoder_inputs, len(char2idx), hp.hidden_units, scope="embed_dec")
             
            # Encoding
            for i in range(hp.num_blocks):
                for rate in (1,2,4,8,16):
                    self.enc = block(self.enc, 
                                    size=5, 
                                    rate=rate,
                                    causal=False,
                                    initial=True if (i==0 and rate==1) else False,
                                    scope="enc_block_{}_{}".format(i, rate)) # (N, T, C)
                     
            # Decoding
            self.dec = tf.concat((self.enc, self.dec), -1)
            for i in range(hp.num_blocks):
                for rate in (1,2,4,8,16):
                        self.dec = block(self.dec, 
                                        size=3, 
                                        rate=rate, 
                                        causal=True,
                                        scope="dec_block_{}_{}".format(i, rate))
             
            # final 1 X 1 convolutional layer for softmax
            self.logits = conv1d(self.dec, filters=len(char2idx), use_bias=True) # (N, T, V)
            
            if is_training:
                # Loss
                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y) # (N, T)
                istarget = tf.to_float(tf.not_equal(self.y, 0)) # zeros: 0, non-zeros: 1 (N, T)
                self.loss = tf.reduce_sum(ce * istarget) / (tf.reduce_sum(istarget) + 1e-8)
                 
                # Training
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_op = tf.train.AdamOptimizer(learning_rate=hp.lr)\
                                        .minimize(self.loss, global_step=self.global_step)
                 
                # Summmary 
                tf.summary.scalar('loss', self.loss)
                self.merged = tf.summary.merge_all()
                
            # Predictions
            self.preds = tf.arg_max(self.logits, dimension=-1)

def main():   
    g = Graph("train"); print("Graph loaded")
    char2idx, idx2char = load_vocab()
    
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)
    
    with sv.managed_session() as sess:
        # Training
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
               
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
        
if __name__ == '__main__':
    main()
    print("Done")

