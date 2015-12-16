__author__ = 'abhishek'

'''
A bi-directional recurrent neural network
Notes:
1. Use sentences instead of words as mini batches for SGD.
2. Use early-stopping on validation set for regularization.


TO FIX:
1. Implement Viterbi decoding using scan
'''

import os
from collections import OrderedDict
import numpy, theano
from theano import tensor as T
from theano import shared, function, scan, printing

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
        self.Wout         = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
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
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

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

    def update_params(self, params_dict):
        self.emb = params_dict['embeddings']
        self.Wx = params_dict['Wx']
        self.Wh_forward = params_dict['Wh_forward']
        self.Wh_backward = params_dict['Wh_backward']
        self.Wc_forward = params_dict['Wc_forward']
        self.Wc_backward = params_dict['Wc_backward']
        self.Wout = params_dict['Wout']
        self.bs_forward = params_dict['bs_forward']
        self.bs_backward = params_dict['bs_backward']
        self.bc = params_dict['bc']
        self.c = params_dict['c']
        self.s0_forward = params_dict['s0_forward']
        self.s0_backward = params_dict['s0_backward']

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

class model_birnn_structured(object):
    def __init__(self, nh, nc, ne, de, cs, decode='greedy'):

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
        self.Wh_forward   = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                            (nh, nh)).astype(theano.config.floatX))
        self.Wh_backward  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                            (nh, nh)).astype(theano.config.floatX))
        self.Wc_forward   = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                            (nh, nh)).astype(theano.config.floatX))
        self.Wc_backward  = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                            (nh, nh)).astype(theano.config.floatX))
        self.Wc_tag       = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                            (nc, nh)).astype(theano.config.floatX))
        self.Wout         = shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                            (nh, nc)).astype(theano.config.floatX))
        self.bs_forward   = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bs_backward  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bc           = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.c            = shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.s0_forward   = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.s0_backward  = shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # Don't make this a parameter, this is the fixed 'START' tag at the beginning of sentence
        self.y0           = shared(numpy.array([0.0] * (nc-1) + [1.0], dtype=theano.config.floatX ))

        # bundle
        self.params = [ self.emb, self.Wx, self.Wh_forward, self.Wh_backward, self.Wc_forward, self.Wc_backward,
                        self.Wout, self.bs_forward, self.bs_backward, self.bc, self.c, self.s0_forward, self.s0_backward]
        self.names  = ['embeddings', 'Wx', 'Wh_forward', 'Wh_backward', 'Wc_forward', 'Wc_backward',
                       'Wout', 'bs_forward', 'bs_backward', 'bc', 'c', 's0_forward', 's0_backward' ]

        #theano.config.compute_test_value = 'off'

        idxs_forward    = T.imatrix() # as many columns as context window size/ lines as words in the sentence
        idxs_backward   = T.imatrix() # as many columns as context window size/ lines as words in the sentence
        #idxs_forward.tag.test_value = numpy.random.randint(low=0, high=9, size=(3, 3)).astype(numpy.int32)
        #idxs_backward.tag.test_value = numpy.random.randint(low=0, high=9, size=(3, 3)).astype(numpy.int32)

        #print idxs_forward.tag
        x_forward       = self.emb[idxs_forward].reshape((idxs_forward.shape[0], de*cs))
        x_backward      = self.emb[idxs_backward].reshape((idxs_backward.shape[0], de*cs))
        y_sentence      = T.ivector('y_sentence') # label

        def recurrence(x_forward_t, x_backward_t, s_forward_tm1, s_backward_tm1, y_tm1):
            s_forward_t = T.nnet.sigmoid(T.dot(x_forward_t, self.Wx) + T.dot(s_forward_tm1, self.Wh_forward)
                                        + self.bs_forward)
            s_backward_t = T.nnet.sigmoid(T.dot(x_backward_t, self.Wx) + T.dot(s_backward_tm1, self.Wh_backward)
                                        + self.bs_backward)
            s_t = T.nnet.sigmoid(T.dot(s_forward_t, self.Wc_forward) + T.dot(s_backward_t, self.Wc_backward)
                                 + T.dot(y_tm1, self.Wc_tag) + self.bc)
            y_t = T.nnet.softmax(T.dot(s_t, self.Wout) + self.c)[0,:]
            return [s_forward_t, s_backward_t, y_t]

        [s_forward, s_backward, y], _ = theano.scan(fn=recurrence, \
                                            sequences=[x_forward, x_backward],
                                            outputs_info=[self.s0_forward, self.s0_backward, self.y0], \
                                            n_steps=x_forward.shape[0])

        # since input is 1 X num_classes
        # first col is word index, next 2 columns are for 2D output of softmax
        #p_y_given_x_lastword = r[-1,0,:]
        #p_y_given_x_sentence = y[:, :]
        p_y_given_x_sentence = y

        def get_trellis_column(p_y_given_x_word, trellis_prev_score):
            trellis_curr_index = []
            trellis_curr_score = []
            for tag_current in xrange(nc):
                best_prev_tag_index = 9
                best_tag_score = -numpy.inf
                for tag_previous in xrange(nc):
                    if T.gt(p_y_given_x_word[tag_current] + trellis_prev_score[tag_previous], best_tag_score):
                        best_prev_tag_index = tag_previous
                        best_tag_score = p_y_given_x_word[tag_current] + trellis_prev_score[tag_previous]
                trellis_curr_index.append(best_prev_tag_index)
                trellis_curr_score.append(best_tag_score)
            return [T.as_tensor_variable(trellis_curr_score), T.as_tensor_variable(trellis_curr_index)]

        if decode == 'greedy':
            y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        else:
            [trellis_score, trellis_index], _ = theano.scan(fn = get_trellis_column, \
                                                outputs_info=[numpy.array([0.0] * nc, theano.config.floatX), None], \
                                                sequences=[p_y_given_x_sentence])
            #printing.debugprint(trellis_score)
            # get the best ending score
            best_index = 0
            back_index = 0
            best_score = -numpy.inf
            for i in xrange(nc):
                if T.gt(trellis_score[-1][i], best_score):
                    best_score = trellis_score[-1][i]
                    #backpointer = trellis[-1][i]
                    back_index = trellis_index[-1][i]
                    best_index = i
            y_pred_tail = [best_index]

            trellis_index = trellis_index[:-1, :]        # remove last column
            trellis_index = trellis_index[::-1, :]       # flip remaining columns
            [back_index, y_pred_head], _ = theano.scan(fn=lambda trellis_column, b_i: [trellis_column[back_index], back_index], \
                                        outputs_info=[back_index, None], \
                                        sequences=[trellis_index])

            y_pred = y_pred_head[::-1] + y_pred_tail
            print y_pred

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

    def update_params(self, params_dict):
        self.emb = params_dict['embeddings']
        self.Wx = params_dict['Wx']
        self.Wh_forward = params_dict['Wh_forward']
        self.Wh_backward = params_dict['Wh_backward']
        self.Wc_forward = params_dict['Wc_forward']
        self.Wc_backward = params_dict['Wc_backward']
        self.Wout = params_dict['Wout']
        self.bs_forward = params_dict['bs_forward']
        self.bs_backward = params_dict['bs_backward']
        self.bc = params_dict['bc']
        self.c = params_dict['c']
        self.s0_forward = params_dict['s0_forward']
        self.s0_backward = params_dict['s0_backward']

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())