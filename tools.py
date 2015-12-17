import random, numpy
import theano
from theano import gof, tensor
from theano.gof import Apply

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [-1] + l + win/2 * [-1]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

class SoftmaxMarginGrad(gof.Op):
    """Gradient wrt x of the SoftmaxMargin Op"""
    nin = 2
    nout = 1

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return tensor.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, dy, sm, **kwargs):
        dy = tensor.as_tensor_variable(dy)
        sm = tensor.as_tensor_variable(sm)
        return Apply(self, [dy, sm], [sm.type.make_variable()])

    def perform(self, node, input_storage, output_storage):
        dy, sm = input_storage
        dx = numpy.zeros_like(sm)
        #dx[i,j] = - (\sum_k dy[i,k] sm[i,k]) sm[i,j] + dy[i,j] sm[i,j]
        for i in xrange(sm.shape[0]):
            dy_times_sm_i = dy[i] * sm[i]
            dx[i] = dy_times_sm_i - sum(dy_times_sm_i) * sm[i]
        output_storage[0][0] = dx

    def grad(self, *args):
        raise NotImplementedError()

    def infer_shape(self, node, shape):
        return [shape[1]]

softmaxmargin_grad = SoftmaxMarginGrad()


class SoftmaxMargin(gof.Op):
    """
    WRITEME
    """

    nin = 1
    nout = 1

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        #p = tensor.as_tensor_variable(p)
        #i = tensor.as_tensor_variable(i)
        if x.type.ndim not in (1, 2) \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats. Got ', x.type)
        if x.ndim == 1:
            x = tensor.shape_padleft(x, n_ones=1)
        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        #x, p, i = input_storage
        x = input_storage
        print 'x', x
        maximum = max(x[0][0])
        e_x = numpy.exp(x - maximum)
        print 'maximum', maximum
        #e_x = numpy.exp(x)
        print 'e_x', e_x
        partition = e_x[0].sum(axis=1)
        print 'partition ', partition
        sm = e_x / partition[:, None]
        print 'sm', sm
        sm[0][0][8] = sm[0][0][8] * partition / (partition + 0.01)
        print 'sm_new', sm
        output_storage = sm

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        sm = softmaxmargin(x)
        #sm = tensor.nnet.softmax(x)
        return [tensor.nnet.softmax_grad(g_sm, sm)]

    def R_op(self, inputs, eval_points):
        pass

    def infer_shape(self, node, shape):
        print shape
        return [shape[0]]

softmaxmargin = SoftmaxMargin()

'''
    def grad(self, inp, grads):
        x,p,i = inp
        g_sm = grads
        #sm = softmaxmargin(x,0.0,8)
        #return [softmaxmargin_grad(g_sm, sm)]
        return x
'''