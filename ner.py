__author__ = 'abhishek'
'''
Named Entity Recognition
The training data consists of the word, POS tag, syntactic chuck tag and an NER tag (the ground truth label)
Output should follow the same structure as input, but also add the predicted output NER tag

'''

from load import CoNLL2k3Loader
import argparse
import os, psutil, sys, subprocess, random, time
import numpy
from collections import OrderedDict, Counter

from rnn import model_rnn
from birnn import model_birnn_unstructured, model_birnn_structured
from lstm import model_lstm
from tools import shuffle, minibatch, contextwin

def generate_data(datatype, word_indexes, class_indexes):
    gen_index = False
    if len(word_indexes) == 0:
        gen_index = True
    words_list = Counter()
    X_sentences= []
    Y_labels = []
    X_idxs = []
    Y_idxs = []

    # read the header
    sentence = []
    loader.get_next_point(sentence, datatype)

    # read all the sentences
    sentence = []
    loader.get_next_point(sentence, datatype)
    sentence_num = 0
    while len(sentence) != 0:
        sentence_num = sentence_num + 1
        if sentence_num > 5:
            break
        window_tokens = loader.get_unwindow_tokens(sentence)
        token_nbr = 0
        input_sequence_words = []
        output_sequence_labels = []
        valid_sentence = True
        for curr in window_tokens:
            word = curr[0]
            if(word == '-DOCSTART-'):
                valid_sentence = False
                break
            # convert numbers
            if word.isdigit():
                word = 'DIGIT' * len(word)
            words_list[word] += 1
            input_sequence_words.append(word)
            output_sequence_labels.append(curr[3])
        if valid_sentence:
            X_sentences.append(input_sequence_words)
            Y_labels.append(output_sequence_labels)
        sentence = []
        loader.get_next_point(sentence, datatype)

    new_word_indexes = {}
    if gen_index:
        # replace infrequent words with UNK, put a switch for this
        final_words_list = set()
        for word in words_list:
            if words_list[word] > 1:
                final_words_list.add(word)
        final_words_list.add('UNK')

        ### generate indexes for all the words and labels
        idx = 0
        for word in final_words_list:
            new_word_indexes[word] = idx
            idx = idx + 1
        word_indexes = new_word_indexes

    ### generate encoded version of the data
    for sentence in X_sentences:
        idxs = []
        for word in sentence:
            if word in word_indexes:
                idxs.append(word_indexes[word])
            else:
                idxs.append(word_indexes['UNK'])
        X_idxs.append(idxs)
    for labels in Y_labels:
        idxs = []
        for label in labels:
            idxs.append(class_indexes[label])
        Y_idxs.append(idxs)

    return X_sentences, Y_labels, X_idxs, Y_idxs, new_word_indexes


parser = argparse.ArgumentParser(description='read the arguments')
parser.add_argument('--train', help='train file')
parser.add_argument('--test', help='test file')
parser.add_argument('--output', help='file to write outputs')
parser.add_argument('--expname', help='name of this experiment configuration')
args = parser.parse_args()

loader = CoNLL2k3Loader(args.train, args.test, 'dummy')

# define the class labels and their inverted index
class_indexes = {}
classes_list = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O', 'START']
class_indexes = {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, \
                'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6, 'I-PER': 7, \
                'O': 8, 'START': 9}

# generate the train and test data in form of context window tokens
X_train_sentences, Y_train_labels, X_train_idxs, Y_train_idxs, word_indexes = generate_data('train', {}, class_indexes)
X_test_sentences, Y_test_labels, X_test_idxs, Y_test_idxs, _ = generate_data('test', word_indexes, class_indexes)

num_embeddings = len(word_indexes)
num_classes = len(class_indexes)
num_train_sentences = len(X_train_idxs)
num_test_sentences = len(X_test_idxs)

#print 'num of train and test sentences: %d, %d' % (num_train_sentences, num_test_sentences)

### Train the model

# parameters for rnn / bi-rnn
s = {'fold':5, # 5 folds 0,1,2,3,4
     'lr':0.0627142536696559,
     'verbose':1,
     'decay':False, # decay on the learning rate if improvement stops
     'win':3, # number of words in the context window
     'bs':9, # number of backprop through time steps            # ???
     'nhidden':100, # number of hidden units
     'seed':345,
     'emb_dimension':100, # dimension of word embedding
     'nepochs':1}

folder = os.path.basename(__file__).split('.')[0] + '-' + args.expname
if not os.path.exists(folder): os.mkdir(folder)

# instantiate the model
numpy.random.seed(s['seed'])
random.seed(s['seed'])
'''
rnn = model( nh = s['nhidden'],
            nc = num_classes,
            ne = num_embeddings,
            de = s['emb_dimension'],
            cs = s['win'] )
'''
'''
birnn = model_birnn_structured( nh = s['nhidden'],
            nc = num_classes,
            ne = num_embeddings,
            de = s['emb_dimension'],
            cs = s['win'],
            decode='greedy')
'''
lstm  = model_lstm( nh = s['nhidden'],
            nc = num_classes,
            ne = num_embeddings,
            de = s['emb_dimension'],
            cs = s['win'] )
best_params = {}

# train with early stopping on validation set
best_f1 = -numpy.inf
s['clr'] = s['lr']
for e in xrange(s['nepochs']):
    # shuffling of data per epoch
    shuffle([X_train_sentences, X_train_idxs, Y_train_labels, Y_train_idxs], s['seed'])
    s['ce'] = e
    tic = time.time()
    for i in xrange(num_train_sentences):
        #print X_train_idxs[i]
        sentence_forward = contextwin(X_train_idxs[i], s['win'])  # words with context in forward direction
        sentence_backward = list(reversed(sentence_forward))
        #print sentence
        #words  = map(lambda x: numpy.asarray(x).astype('int32'),\
        #             minibatch(sentence, s['bs']))        # batch them up, why ??
        labels = Y_train_idxs[i]
        #for word_batch , label_last_word in zip(words, labels):
        #rnn.sentence_train(sentence_forward, labels, s['clr'])
        #rnn.normalize()
        #birnn.sentence_train(sentence_forward, sentence_backward, labels, s['clr'])
        #birnn.normalize()
        lstm.sentence_train(sentence_forward, labels, s['clr'])
        lstm.normalize()
        if s['verbose']:
            print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./num_train_sentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
            sys.stdout.flush()

    # evaluation
    total_tokens_predicted = 0
    correct_tokens_predicted = 0
    total_tokens_gold = 0
    total_tags = 0
    correct_tags = 0
    for i in xrange(num_test_sentences):
        sentence_forward = contextwin(X_test_idxs[i], s['win'])
        sentence_backward = list(reversed(sentence_forward))
        ground_truth_labels = numpy.asarray(Y_test_idxs[i])
        #predicted_labels = birnn.sentence_classify(sentence_forward, sentence_backward)
        predicted_labels = lstm.sentence_classify(sentence_forward)
        #print ground_truth_labels
        #print predicted_labels
        total_tags += len(ground_truth_labels)
        correct_tags += sum(ground_truth_labels == predicted_labels)
        correct_tokens_predicted += sum([1 if (x != 8 and x == y) else 0
                                         for (x,y) in zip(ground_truth_labels, predicted_labels)])
        total_tokens_gold += sum([1 if x != 8 else 0 for x in ground_truth_labels])
        total_tokens_predicted += sum([1 if x != 8 else 0 for x in predicted_labels])

    print correct_tokens_predicted, total_tokens_gold, total_tokens_predicted, total_tags, correct_tags
    accuracy = float(correct_tags) / float(total_tags)
    precision = float(correct_tokens_predicted) / float(total_tokens_predicted) if total_tokens_predicted > 0 else 0.0
    recall = float(correct_tokens_predicted) / float(total_tokens_gold) if total_tokens_gold > 0 else 0.0
    f1score = float(2.0 * precision * recall) / float(precision + recall) if precision + recall > 0 else 0.0
    print 'epoch %d: accuracy=%.4f, precision= %.4f, recall=%.4f, f1= %.4f' % (e, accuracy, precision, recall, f1score)
    if f1score > best_f1:
        for param, name in zip(lstm.params, lstm.names):
            best_params[name] = param
        best_f1 = f1score
        if s['verbose']:
            print 'NEW BEST: epoch', e, 'valid F1', f1score, 'best test F1', f1score, ' '*20
        s['vf1'], s['vp'], s['vr'] = f1score, precision, recall
        s['tf1'], s['tp'], s['tr'] = f1score, precision,  recall
        s['be'] = e
    else:
        print ''

    #Wx = numpy.load(os.path.join('ner_2', 'Wx' + '.npy'))
    #print Wx

    # learning rate decay if no improvement in 10 epochs
    if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5
    if s['clr'] < 1e-5: break

print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder

# best model parameters have been saved, update the model with these and save these to a folder
lstm.update_params(best_params)
lstm.save(folder)
#loader.close_files()

# write the predictions to file
X_idxs = []
Y_idxs = []
prediction_loader = CoNLL2k3Loader('dummy', args.test, args.output + '-' + args.expname)

# read the header
sentence = []
prediction_loader.get_next_point(sentence, 'test')
prediction_loader.write_line_tokens(sentence)

# read all the sentences, a sentence is a list of lists
sentence = []
prediction_loader.get_next_point(sentence, 'test')
while len(sentence) != 0:
    window_tokens = prediction_loader.get_unwindow_tokens(sentence)        # list of line tuples
    #token_nbr = 0
    input_sequence_words = []
    valid_sentence = True
    for curr in window_tokens:
        word = curr[0]
        if(word == '-DOCSTART-'):
            valid_sentence = False
            break
        # convert numbers
        if word.isdigit():
            word = 'DIGIT' * len(word)
        input_sequence_words.append(word)

    if valid_sentence:
        idxs = []
        for word in input_sequence_words:
            if word in word_indexes:
                idxs.append(word_indexes[word])
            else:
                idxs.append(word_indexes['UNK'])
        test_sentence_forward = contextwin(idxs, s['win'])
        test_sentence_backward = list(reversed(test_sentence_forward))
        #print len(test_sentence_forward), len(test_sentence_backward)
        #test_labels = birnn.sentence_classify(test_sentence_forward, test_sentence_backward)
        test_labels = lstm.sentence_classify(test_sentence_forward)
        test_tokens = [classes_list[label] for label in test_labels]
        prediction_loader.write_output_tokens(test_tokens, sentence)
    else:
        prediction_loader.write_line_tokens(sentence)

    sentence = []
    prediction_loader.get_next_point(sentence, 'test')