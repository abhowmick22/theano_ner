__author__ = 'abhishek'

'''
A recurrent neural network
Notes:
1. Use sentences instead of words as mini batches for SGD.
2. Use early-stopping on validation set for regularization.
3. Uses words, pos and chunk tags as features
'''

import os
from collections import OrderedDict
import numpy, theano
from theano import tensor as T
from theano import shared, function, scan
import tools

class model_rnn(object):
    def __init__(self, nh, nc, ne, np, nch, de, dp, dch, cs, mp):

        # nh :: dimension of the hidden layer
        # nc :: number of classes
        # ne :: number of word embeddings in the vocabulary
        # np :: number of pos tags in the vocabulary
        # nch:: number of chunk tags in the vocabulary
        # de :: dimension of the word embeddings
        # dp :: dimension of the pos tag embeddings
        # dch:: dimension of the chunk embeddings
        # cs :: word window context size
        # mp :: margin penalty, for the soft-max margin

        # parameters of the model, add one for PADDING at the end (corresponds to -1)
        self.emb = shared(0.2 * numpy.random.uniform(-1.0, 1.0, \
                   (ne+1, de)).astype(theano.config.floatX))
        self.pos_emb = shared(0.2 * numpy.random.uniform(-1.0, 1.0, \
                   (np+1, dp)).astype(theano.config.floatX))
        self.chunk_emb = shared(0.2 * numpy.random.uniform(-1.0, 1.0, \
                   (nch+1, dch)).astype(theano.config.floatX))
        self.Wx       = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wpos    = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dp * cs, nh)).astype(theano.config.floatX))
        self.Wchunk  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dch * cs, nh)).astype(theano.config.floatX))
        self.Wh  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.Wout   = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bs  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.c   = shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.s0  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # not a param
        self.sm_bias = shared(numpy.array([1.0] * (nc-2) + [1.0 / mp] + [1.0], dtype=theano.config.floatX ))

        # bundle
        self.params = [ self.emb, self.pos_emb, self.chunk_emb, self.Wx, self.Wpos, self.Wchunk, self.Wh, self.Wout, self.bs, self.c, self.s0 ]
        self.names  = ['embeddings', 'pos_embeddings', 'chunk_embeddings', 'Wx', 'Wpos', 'Wchunk', 'Wh', 'Wout', 'bs', 'c', 's0']
        idxs = T.imatrix() # as many columns as context window size/ lines as words in the sentence
        pos_idxs = T.imatrix()
        chunk_idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        pos = self.pos_emb[pos_idxs].reshape((pos_idxs.shape[0], dp*cs))
        chunks = self.chunk_emb[chunk_idxs].reshape((chunk_idxs.shape[0], dch*cs))
        y_sentence = T.ivector('y_sentence') # label
        #softmax_margin = tools.SoftmaxMargin()

        def recurrence(x_t, p_t, ch_t, s_tm1):
            s_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(p_t, self.Wpos) +
                                 T.dot(ch_t, self.Wchunk) + T.dot(s_tm1, self.Wh) + self.bs)
            r_t = T.nnet.softmax(T.dot(s_t, self.Wout) + self.c) * self.sm_bias
            #r_t = softmax_with_bias(T.dot(s_t, self.Wout) + self.c, self.sm_bias)
            return [s_t, r_t]

        [s, r], _ = theano.scan(fn=recurrence, \
                    sequences=[x, pos, chunks], outputs_info=[self.s0, None], \
                    n_steps=x.shape[0])

        # since input is 1 X num_classes
        # first col is word index, next 2 columns are for 2D output of softmax
        p_y_given_x_lastword = r[-1,0,:]
        p_y_given_x_sentence = r[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1) # argmax for each position in sentence

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        # this is the loss
        sentence_nll = -T.sum(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])     # assignment instructs to use sum and not mean
        sentence_gradients = T.grad( sentence_nll, self.params )
        sentence_updates = OrderedDict(( p, p-lr*g )
                                       for p, g in
                                       zip( self.params , sentence_gradients))

        # theano functions
        self.sentence_classify = theano.function(inputs=[idxs, pos_idxs, chunk_idxs], outputs=y_pred)
        self.sentence_train = theano.function( inputs = [idxs, pos_idxs, chunk_idxs, y_sentence, lr],
                                      outputs = sentence_nll,
                                      updates = sentence_updates )

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb / T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def update_params(self, params_dict):
        self.emb = params_dict['embeddings']
        self.pos_emb = params_dict['pos_embeddings']
        self.chunk_emb = params_dict['chunk_embeddings']
        self.Wx = params_dict['Wx']
        self.Wpos = params_dict['Wpos']
        self.Wchunk = params_dict['Wchunk']
        self.Wh = params_dict['Wh']
        self.Wout = params_dict['Wout']
        self.bs = params_dict['bs']
        self.c = params_dict['c']
        self.s0 = params_dict['s0']

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())