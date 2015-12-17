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

def generate_data(datatype, word_indexes, pos_indexes, chunk_indexes, class_indexes, additional_features):
    gen_index = False
    if len(word_indexes) == 0:
        gen_index = True
    words_list = Counter()
    pos_list = Counter()
    chunk_list = Counter()
    X_sentences= []
    X_pos = []
    X_chunk = []
    Y_labels = []
    X_idxs = []
    X_pos_idxs = []
    X_chunk_idxs = []
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
        if sentence_num > 100:
            break
        window_tokens = loader.get_unwindow_tokens(sentence)
        #token_nbr = 0
        input_sequence_words = []
        input_sequence_pos = []
        input_sequence_chunk = []
        output_sequence_labels = []
        valid_sentence = True
        for curr in window_tokens:
            word, pos, chunk, label = curr
            if(word == '-DOCSTART-'):
                valid_sentence = False
                break
            # convert numbers
            if word.isdigit():
                word = 'DIGIT' * len(word)
            words_list[word] += 1
            pos_list[pos] += 1
            chunk_list[chunk] += 1
            input_sequence_words.append(word)
            input_sequence_pos.append(pos)
            input_sequence_chunk.append(chunk)
            output_sequence_labels.append(label)
        if valid_sentence:
            X_sentences.append(input_sequence_words)
            X_pos.append(input_sequence_pos)
            X_chunk.append(input_sequence_chunk)
            Y_labels.append(output_sequence_labels)
        sentence = []
        loader.get_next_point(sentence, datatype)

    new_word_indexes = {}
    new_pos_indexes = {}
    new_chunk_indexes = {}

    if gen_index:
        # replace infrequent words with UNK, put a switch for this
        final_words_list = set()
        for word in words_list:
            if words_list[word] > 1:
                final_words_list.add(word)
        final_words_list.add('UNK')
        final_pos_list = set()
        for pos in pos_list:
            if pos_list[pos] > 1:
                final_pos_list.add(pos)
        final_pos_list.add('UNK')
        final_chunk_list = set()
        for chunk in chunk_list:
            if chunk_list[chunk] > 1:
                final_chunk_list.add(chunk)
        final_chunk_list.add('UNK')

        ### generate indexes for all the words and labels
        idx = 0
        for word in final_words_list:
            new_word_indexes[word] = idx
            idx = idx + 1
        idx = 0
        for pos in final_pos_list:
            new_pos_indexes[pos] = idx
            idx = idx + 1
        idx = 0
        for chunk in final_chunk_list:
            new_chunk_indexes[chunk] = idx
            idx = idx + 1
        word_indexes = new_word_indexes
        pos_indexes = new_pos_indexes
        chunk_indexes = new_chunk_indexes

    ### generate encoded version of the data
    for sentence in X_sentences:
        idxs = []
        for word in sentence:
            if word in word_indexes:
                idxs.append(word_indexes[word])
            else:
                idxs.append(word_indexes['UNK'])
        X_idxs.append(idxs)
    for sentence_pos in X_pos:
        idxs = []
        for pos in sentence_pos:
            if pos in pos_indexes and additional_features:
                idxs.append(pos_indexes[pos])
            else:
                idxs.append(pos_indexes['UNK'])
        X_pos_idxs.append(idxs)
    for sentence_chunk in X_chunk:
        idxs = []
        for chunk in sentence_chunk:
            if chunk in chunk_indexes and additional_features:
                idxs.append(chunk_indexes[chunk])
            else:
                idxs.append(chunk_indexes['UNK'])
        X_chunk_idxs.append(idxs)
    for labels in Y_labels:
        idxs = []
        for label in labels:
            idxs.append(class_indexes[label])
        Y_idxs.append(idxs)

    return X_idxs, X_pos_idxs, X_chunk_idxs, Y_idxs, new_word_indexes, new_pos_indexes, new_chunk_indexes


parser = argparse.ArgumentParser(description='read the arguments')
parser.add_argument('--train', help='train file')
parser.add_argument('--val', help='val file')
parser.add_argument('--test', help='test file')
parser.add_argument('--features', dest='features', action='store_true', help='whether to use additional features')
parser.add_argument('--expname', help='name of this experiment configuration')
parser.set_defaults(features=True)
args = parser.parse_args()

loader = CoNLL2k3Loader(args.train, args.val, 'dummy')

# define the class labels and their inverted index
class_indexes = {}
classes_list = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O', 'START']
class_indexes = {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, \
                'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6, 'I-PER': 7, \
                'O': 8, 'START': 9}

# generate the train and test data in form of context window tokens
X_train_idxs, X_train_pos_idxs, X_train_chunk_idxs, Y_train_idxs, \
    word_indexes, pos_indexes, chunk_indexes = generate_data('train', {}, {}, {}, class_indexes, args.features)
X_test_idxs, X_test_pos_idxs, X_test_chunk_idxs, Y_test_idxs, \
    _, _, _ = generate_data('test', word_indexes, pos_indexes, chunk_indexes, class_indexes, args.features)

num_embeddings = len(word_indexes)
num_pos_embeddings = len(pos_indexes)
num_chunk_embeddings = len(chunk_indexes)
num_classes = len(class_indexes)
num_train_sentences = len(X_train_idxs)
num_test_sentences = len(X_test_idxs)

### Train the model

# parameters for rnn / bi-rnn
s = {'fold':5, # 5 folds 0,1,2,3,4
     'lr':0.01,
     'verbose':1,
     'decay':False, # decay on the learning rate if improvement stops
     'win':3, # number of words in the context window
     'bs':9, # number of backprop through time steps            # ???
     'nhidden':100, # number of hidden units
     'seed':345,
     'emb_dimension':100, # dimension of word embedding
     'pos_emb_dimension':5, # dimension of pos embedding
     'chunk_emb_dimension':5, # dimension of chunk embedding
     'nepochs':2}

folder = os.path.basename(__file__).split('.')[0] + '-' + args.expname
if not os.path.exists(folder): os.mkdir(folder)

# instantiate the model
numpy.random.seed(s['seed'])
random.seed(s['seed'])

rnn = model_rnn( nh = s['nhidden'],
            nc = num_classes,
            ne = num_embeddings,
            np = num_pos_embeddings,
            nch = num_chunk_embeddings,
            de = s['emb_dimension'],
            dp = s['pos_emb_dimension'],
            dch = s['chunk_emb_dimension'],
            cs = s['win'],
            mp = 10.0)

'''
birnn = model_birnn_structured( nh = s['nhidden'],
            nc = num_classes,
            ne = num_embeddings,
            de = s['emb_dimension'],
            cs = s['win'],
            decode='greedy')
'''
'''
lstm  = model_lstm( nh = s['nhidden'],
            nc = num_classes,
            ne = num_embeddings,
            de = s['emb_dimension'],
            cs = s['win'] )
'''

best_params = {}
# train with early stopping on validation set
best_f1 = -numpy.inf
s['clr'] = s['lr']
training_loss = []
for e in xrange(s['nepochs']):
    # shuffling of data per epoch
    shuffle([X_train_idxs, X_train_pos_idxs, X_train_chunk_idxs, Y_train_idxs], s['seed'])
    s['ce'] = e
    tic = time.time()
    loss = 0.0
    for i in xrange(num_train_sentences):
        #print X_train_idxs[i]
        sentence_forward = contextwin(X_train_idxs[i], s['win'])
        sentence_backward = list(reversed(sentence_forward))
        sentence_pos_forward = contextwin(X_train_pos_idxs[i], s['win'])
        sentence_pos_backward = list(reversed(sentence_pos_forward))
        sentence_chunk_forward = contextwin(X_train_chunk_idxs[i], s['win'])
        sentence_chunk_backward = list(reversed(sentence_chunk_forward))
        labels = Y_train_idxs[i]
        loss += rnn.sentence_train(sentence_forward, sentence_pos_forward, sentence_chunk_forward, labels, s['clr'])
        rnn.normalize()
        #birnn.sentence_train(sentence_forward, sentence_backward, labels, s['clr'])
        #birnn.normalize()
        #lstm.sentence_train(sentence_forward, labels, s['clr'])
        #lstm.normalize()
        if s['verbose']:
            print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./num_train_sentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
            sys.stdout.flush()
    loss = round(loss /  float(num_train_sentences), 4)
    training_loss.append(loss)
    # evaluation
    total_tokens_predicted = 0
    correct_tokens_predicted = 0
    total_tokens_gold = 0
    total_tags = 0
    correct_tags = 0
    for i in xrange(num_test_sentences):
        sentence_forward = contextwin(X_test_idxs[i], s['win'])
        sentence_backward = list(reversed(sentence_forward))
        sentence_pos_forward = contextwin(X_test_pos_idxs[i], s['win'])
        sentence_pos_backward = list(reversed(sentence_pos_forward))
        sentence_chunk_forward = contextwin(X_test_chunk_idxs[i], s['win'])
        sentence_chunk_backward = list(reversed(sentence_chunk_forward))
        ground_truth_labels = numpy.asarray(Y_test_idxs[i])
        predicted_labels = rnn.sentence_classify(sentence_forward, sentence_pos_forward, sentence_chunk_forward)
        #predicted_labels = birnn.sentence_classify(sentence_forward, sentence_backward)
        #predicted_labels = lstm.sentence_classify(sentence_forward)
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
        for param, name in zip(rnn.params, rnn.names):
            best_params[name] = param
        best_f1 = f1score
        if s['verbose']:
            print 'NEW BEST: epoch', e, 'valid F1', f1score, 'best test F1', f1score, ' '*20
        s['vf1'], s['vp'], s['vr'] = f1score, precision, recall
        s['tf1'], s['tp'], s['tr'] = f1score, precision,  recall
        s['be'] = e
    else:
        print ''

    # learning rate decay if no improvement in 10 epochs
    if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5
    if s['clr'] < 1e-5: break

print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder

# log the training curve info
training_curve = open('training_loss' + '-' + args.expname, 'w')
training_curve.write(str(training_loss))
training_curve.close()

# best model parameters have been saved, update the model with these and save these to a folder
rnn.update_params(best_params)
rnn.save(folder)
#loader.close_files()

# write the predictions to file
X_idxs = []
X_pos_idxs = []
X_chunk_idxs = []
Y_idxs = []
prediction_loader = CoNLL2k3Loader('dummy', args.test, 'output-' + args.expname)

# read the header
#sentence = []
#prediction_loader.get_next_point(sentence, 'test')
#prediction_loader.write_line_tokens(sentence)

# read all the sentences, a sentence is a list of lists
sentence = []
prediction_loader.get_next_point(sentence, 'test')
while len(sentence) != 0:
    window_tokens = prediction_loader.get_unwindow_tokens(sentence)        # list of line tuples
    input_sequence_words = []
    input_sequence_pos = []
    input_sequence_chunk = []
    valid_sentence = True
    for curr in window_tokens:
        word, pos, chunk = curr
        if(word == '-DOCSTART-'):
            valid_sentence = False
            break
        # convert numbers
        if word.isdigit():
            word = 'DIGIT' * len(word)
        input_sequence_words.append(word)
        input_sequence_pos.append(pos)
        input_sequence_chunk.append(chunk)
    if valid_sentence:
        idxs = []
        pos_idxs = []
        chunk_idxs = []
        for word in input_sequence_words:
            if word in word_indexes:
                idxs.append(word_indexes[word])
            else:
                idxs.append(word_indexes['UNK'])
        for pos in input_sequence_pos:
            if pos in pos_indexes and args.features:
                pos_idxs.append(pos_indexes[pos])
            else:
                pos_idxs.append(pos_indexes['UNK'])
        for chunk in input_sequence_chunk:
            if chunk in chunk_indexes and args.features:
                chunk_idxs.append(chunk_indexes[chunk])
            else:
                chunk_idxs.append(chunk_indexes['UNK'])
        test_sentence_forward = contextwin(idxs, s['win'])
        test_sentence_backward = list(reversed(test_sentence_forward))
        test_sentence_pos_forward = contextwin(pos_idxs, s['win'])
        test_sentence_pos_backward = list(reversed(test_sentence_pos_forward))
        test_sentence_chunk_forward = contextwin(chunk_idxs, s['win'])
        test_sentence_chunk_backward = list(reversed(test_sentence_chunk_forward))
        test_labels = rnn.sentence_classify(test_sentence_forward, test_sentence_pos_forward, test_sentence_chunk_forward)
        #test_labels = birnn.sentence_classify(test_sentence_forward, test_sentence_backward)
        #test_labels = lstm.sentence_classify(test_sentence_forward)
        test_tokens = [classes_list[label] for label in test_labels]
        prediction_loader.write_output_tokens(test_tokens, sentence)
    else:
        prediction_loader.write_line_tokens(sentence)

    sentence = []
    prediction_loader.get_next_point(sentence, 'test')