__author__ = 'abhishek'

'''
A bi-directional recurrent neural network
Notes:
1. Use sentences instead of words as mini batches for SGD.
2. Use early-stopping on validation set for regularization.
'''

import os
from collections import OrderedDict
import numpy, theano
from theano import tensor as T
from theano import shared, function, scan

class model_birnn_unstructured(object):
    def __init__(self, nh, nc, ne, de, cs):

        # nh :: dimension of the hidden layer
        # nc :: number of classes
        # ne :: number of word embeddings in the vocabulary
        # de :: dimension of the word embeddings
        # cs :: word window context size

        # parameters of the model
        self.emb = shared(0.2 * numpy.random.uniform(-1.0, 1.0, \
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end (corresponds to -1)
        self.Wx  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wh_forward  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                            (nh, nh)).astype(theano.config.floatX))
        self.Wh_backward  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                            (nh, nh)).astype(theano.config.floatX))
        self.Wc_forward  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                            (nh, nh)).astype(theano.config.floatX))
        self.Wc_backward  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                            (nh, nh)).astype(theano.config.floatX))
        self.Wout   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bs_forward  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bs_backward  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bc  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.c   = shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.s0_forward  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.s0_backward  = shared(numpy.zeros(nh, dtype=theano.config.floatX))

        # bundle
        self.params = [ self.emb, self.Wx, self.Wh_forward, self.Wh_backward, self.Wc_forward, self.Wc_backward,
                        self.Wout, self.bs_forward, self.bs_backward, self.bc, self.c, self.s0_forward, self.s0_backward]
        self.names  = ['embeddings', 'Wx', 'Wh_forward', 'Wh_backward', 'Wc_forward', 'Wc_backward',
                       'Wout', 'bs_forward', 'bs_backward', 'bc', 'c', 's0_forward', 's0_backward' ]
        idxs_forward = T.imatrix() # as many columns as context window size/ lines as words in the sentence
        idxs_backward = T.imatrix() # as many columns as context window size/ lines as words in the sentence
        x_forward = self.emb[idxs_forward].reshape((idxs_forward.shape[0], de*cs))
        x_backward = self.emb[idxs_backward].reshape((idxs_backward.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence') # label

        def recurrence(x_forward_t, x_backward_t, s_forward_tm1, s_backward_tm1):
            s_forward_t = T.nnet.sigmoid(T.dot(x_forward_t, self.Wx) + T.dot(s_forward_tm1, self.Wh_forward)
                                        + self.bs_forward)
            s_backward_t = T.nnet.sigmoid(T.dot(x_backward_t, self.Wx) + T.dot(s_backward_tm1, self.Wh_backward)
                                        + self.bs_backward)
            #print type(s_forward_t), len(s_forward_t), type(s_backward_t), len(s_backward_t)
            s_t = T.nnet.sigmoid(T.dot(s_forward_t, self.Wc_forward) + T.dot(s_backward_t, self.Wc_backward) + self.bc)
            r_t = T.nnet.softmax(T.dot(s_t, self.Wout) + self.c)
            return [s_forward_t, s_backward_t, r_t]

        [s_forward, s_backward, r], _ = theano.scan(fn=recurrence, \
                                            sequences=[x_forward, x_backward],
                                            outputs_info=[self.s0_forward, self.s0_backward, None], \
                                            n_steps=x_forward.shape[0])

        # since input is 1 X num_classes
        # first col is word index, next 2 columns are for 2D output of softmax
        #p_y_given_x_lastword = r[-1,0,:]
        p_y_given_x_sentence = r[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1) # argmax for each position in sentence

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        sentence_nll = -T.sum(T.log(p_y_given_x_sentence)
                               [T.arange(x_forward.shape[0]), y_sentence])     # assignment instructs to use sum and not mean
        sentence_gradients = T.grad( sentence_nll, self.params )
        sentence_updates = OrderedDict(( p, p-lr*g )
                                       for p, g in
                                       zip( self.params , sentence_gradients))

        # theano functions
        self.sentence_classify = theano.function(inputs=[idxs_forward, idxs_backward], outputs=y_pred)
        self.sentence_train = theano.function( inputs = [idxs_forward, idxs_backward, y_sentence, lr],
                                      outputs = sentence_nll,
                                      updates = sentence_updates )

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb / T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())