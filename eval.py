#-*- coding: utf-8-*-
from __future__ import print_function
import codecs
import sugartensor as tf
import numpy as np
from prepro import *
from train import Graph
from nltk.translate.bleu_score import corpus_bleu

def eval(): 
    # Load graph
    g = Graph(mode="test")
        
    with tf.Session() as sess:
        # Initialize variables
        tf.sg_init(sess)

        # Restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
        print("Restored!")
        mname = open('asset/train/checkpoint', 'r').read().split('"')[1] # model name
        
        # Load data
        X, Sources, Targets = load_test_data()
        char2idx, idx2char = load_vocab()
        
        with codecs.open("results.txt", "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []
            for i in range(len(X) // Hp.bs):
                # Get mini-batches
                x = X[i*Hp.bs: (i+1)*Hp.bs] # mini-batch
                sources = Sources[i*Hp.bs: (i+1)*Hp.bs]
                targets = Targets[i*Hp.bs: (i+1)*Hp.bs]
                
                preds_prev = np.zeros((Hp.bs, Hp.maxlen), np.int32)
                preds = np.zeros((Hp.bs, Hp.maxlen), np.int32)        
                for j in range(Hp.maxlen):
                    # predict next character
                    outs = sess.run(g.preds, {g.x: x, g.y_src: preds_prev})
                    # update character sequence
                    if j < Hp.maxlen - 1:
                        preds_prev[:, j + 1] = outs[:, j]
                    preds[:, j] = outs[:, j]
                
                # Write to file
                for source, target, pred in zip(sources, targets, preds): # sentence-wise
                    got = "".join(idx2char[idx] for idx in pred).split(u"␃")[0]
                    fout.write("- source: " + source +"\n")
                    fout.write("- expected: " + target + "\n")
                    fout.write("- got: " + got + "\n\n")
                    fout.flush()
                    
                    # For bleu score
                    ref = target.split()
                    hypothesis = got.split()
                    if len(ref) > 2:
                        list_of_refs.append([ref])
                        hypotheses.append(hypothesis)
            
            # Get bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            fout.write("Bleu Score = " + str(100*score))
                                            
if __name__ == '__main__':
    eval()
    print("Done")
    
    
