__author__ = 'abhishek'

__author__ = 'abhishek'

'''
An LSTM, without dropout
Notes:
Currently uses greedy decoding


TO FIX:
1. Implement Viterbi decoding using scan
'''

import os
from collections import OrderedDict
import numpy, theano
from theano import tensor as T
from theano import shared, function, scan, printing

class model_lstm(object):
    def __init__(self, nh, nc, ne, np, nch, de, dp, dch, cs, mp):

        # nh :: dimension of the hidden layer
        # nc :: number of classes
        # ne :: number of word embeddings in the vocabulary
        # np :: number of pos embeddings in the vocabulary
        # nch :: number of chunk embeddings in the vocabulary
        # de :: dimension of the word embeddings
        # dp :: dimension of the pos embeddings
        # dch :: dimension of the chunk embeddings
        # cs :: word window context size
        # mp :: margin penalty, for the soft-max margin

        # parameters of the model
        self.emb = shared(0.2 * numpy.random.uniform(-1.0, 1.0, \
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end (corresponds to -1)
        self.pos_emb = shared(0.2 * numpy.random.uniform(-1.0, 1.0, \
                   (np+1, dp)).astype(theano.config.floatX))
        self.chunk_emb = shared(0.2 * numpy.random.uniform(-1.0, 1.0, \
                   (nch+1, dch)).astype(theano.config.floatX))

        # Weight matrices for input
        self.Wi  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wf  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wc  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wo  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))

        # Weight matrices for pos input
        self.Wipos  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dp * cs, nh)).astype(theano.config.floatX))
        self.Wfpos  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dp * cs, nh)).astype(theano.config.floatX))
        self.Wcpos  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dp * cs, nh)).astype(theano.config.floatX))
        self.Wopos  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dp * cs, nh)).astype(theano.config.floatX))

        # Weight matrices for chunk input
        self.Wichunk  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dch * cs, nh)).astype(theano.config.floatX))
        self.Wfchunk  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dch * cs, nh)).astype(theano.config.floatX))
        self.Wcchunk  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dch * cs, nh)).astype(theano.config.floatX))
        self.Wochunk  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dch * cs, nh)).astype(theano.config.floatX))

        # Weight matrices for the representation sequence h_t
        self.Ui  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.Uf  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.Uc  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.Uo  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))

        # Weight matrix for the state value, C_t
        self.Vo  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))

        # Bias vectors
        self.bi  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bf  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bc  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bo  = shared(numpy.zeros(nh, dtype=theano.config.floatX))

        # Weight and bias params for the final soft-max
        self.Wout = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.c   = shared(numpy.zeros(nc, dtype=theano.config.floatX))

        # Initial state of the trained LSTM - state and representation sequence of the memory cell
        self.C0  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.h0  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # not a param
        self.sm_bias = shared(numpy.array([1.0] * (nc-2) + [1.0 / mp] + [1.0], dtype=theano.config.floatX ))

        # bundle
        self.params = [ self.emb, self.pos_emb, self.chunk_emb, self.Wi, self.Wf, self.Wc, self.Wo,
                        self.Wipos, self.Wfpos, self.Wcpos, self.Wopos,
                        self.Wichunk, self.Wfchunk, self.Wcchunk, self.Wochunk,
                        self.Ui, self.Uf, self.Uc, self.Uo,
                        self.bi, self.bf, self.bc, self.bo, self.Wout, self.c, self.C0, self.h0]
        self.names  = ['embeddings', 'pos_embeddings', 'chunk_embeddings', 'Wi', 'Wf', 'Wc', 'Wo', 'Wipos', 'Wfpos', 'Wcpos', 'Wopos',
                       'Wichunk', 'Wfchunk', 'Wcchunk', 'Wochunk',
                       'Ui', 'Uf', 'Uc', 'Uo',
                       'bi', 'bf', 'bc', 'bo', 'Wout', 'c', 'C0', 'h0' ]
        idxs = T.imatrix() # as many columns as context window size/ lines as words in the sentence
        pos_idxs = T.imatrix()
        chunk_idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        pos = self.pos_emb[pos_idxs].reshape((pos_idxs.shape[0], dp*cs))
        chunks = self.chunk_emb[chunk_idxs].reshape((chunk_idxs.shape[0], dch*cs))
        y_sentence = T.ivector('y_sentence') # label

        def recurrence(x_t, p_t, ch_t, C_tm1, h_tm1):

            i_t = T.nnet.sigmoid(T.dot(x_t, self.Wi) + T.dot(p_t, self.Wipos) + T.dot(ch_t, self.Wichunk) + T.dot(h_tm1, self.Ui) + self.bi)     # the input gate
            C_cand_t = T.tanh(T.dot(x_t, self.Wc) + T.dot(p_t, self.Wcpos) + T.dot(ch_t, self.Wcchunk) + T.dot(h_tm1, self.Uc) + self.bc)        # candidate value for state
            f_t = T.nnet.sigmoid(T.dot(x_t, self.Wf) + T.dot(p_t, self.Wfpos) + T.dot(ch_t, self.Wfchunk) + T.dot(h_tm1, self.Uf) + self.bf)     # forget gate activation
            C_t = i_t * C_cand_t + f_t * C_tm1                                              # memory cell state
            o_t = T.nnet.sigmoid(T.dot(x_t, self.Wo) + T.dot(p_t, self.Wopos) + T.dot(ch_t, self.Wochunk) +
                                 T.dot(h_tm1, self.Uo) +              # the output gate
                                 T.dot(C_t, self.Vo) + self.bo)
            h_t = o_t * T.tanh(C_t)                                                         # the cell output
            r_t = T.nnet.softmax(T.dot(h_t, self.Wout) + self.c)                            # the soft-max output
            return [C_t, h_t, r_t]

        [C, h, r], _ = theano.scan(fn=recurrence, \
                                            sequences=[x, pos, chunks],
                                            outputs_info=[self.C0, self.h0, None], \
                                            n_steps=x.shape[0])

        # since input is 1 X num_classes
        # first col is word index, next 2 columns are for 2D output of softmax
        p_y_given_x_sentence = r[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
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
        self.Wi = params_dict['Wi']
        self.Wf = params_dict['Wf']
        self.Wc = params_dict['Wc']
        self.Wo = params_dict['Wo']
        self.Wipos = params_dict['Wipos']
        self.Wfpos = params_dict['Wfpos']
        self.Wcpos = params_dict['Wcpos']
        self.Wopos = params_dict['Wopos']
        self.Wichunk = params_dict['Wichunk']
        self.Wfchunk = params_dict['Wfchunk']
        self.Wcchunk = params_dict['Wcchunk']
        self.Wochunk = params_dict['Wochunk']
        self.Ui = params_dict['Ui']
        self.Uf = params_dict['Uf']
        self.Uc = params_dict['Uc']
        self.Uo = params_dict['Uo']
        self.bi = params_dict['bi']
        self.bf = params_dict['bf']
        self.bc = params_dict['bc']
        self.bo = params_dict['bo']
        self.Wout = params_dict['Wout']
        self.c = params_dict['c']
        self.C0 = params_dict['C0']
        self.h0 = params_dict['h0']

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
