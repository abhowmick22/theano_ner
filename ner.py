__author__ = 'abhishek'
'''
A recurrent neural network (Elman network) for Named Entity Recognition
The data consists of the word, POS tag, syntactic chuck tag and a final NER tag (the label)
'''

from load import CoNLL2k3Loader
import argparse
import os, psutil
import numpy
from collections import OrderedDict, Counter

process = psutil.Process(os.getpid())
parser = argparse.ArgumentParser(description='read the arguments')
parser.add_argument('--train', help='train file')
parser.add_argument('--test', help='test file')
parser.add_argument('--output', help='file to write outputs')
args = parser.parse_args()

# generate the training data in form of context window tokens
#classes_list = set()
words_list = Counter()
X_train_sentences= []
Y_train_labels = []
X_train_idxs = []
Y_train_idxs = []

loader = CoNLL2k3Loader(args.train, args.test, args.output)

# read the header
sentence = []
loader.get_next_point(sentence, 'train')

# read all the sentences
sentence = []
loader.get_next_point(sentence, 'train')
sentence_num = 0
while len(sentence) != 0:
    sentence_num = sentence_num + 1
    #if sentence_num > 5:
    #    break
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
        X_train_sentences.append(input_sequence_words)
        Y_train_labels.append(output_sequence_labels)
        '''
        if sentence_num % 1000 == 0:
            print 'sentence %d ' % sentence_num
            print process.get_memory_info()
        '''
    sentence = []
    loader.get_next_point(sentence, 'train')

# replace infrequent words with UNK, put a switch for this
final_words_list = set()
for word in words_list:
    if words_list[word] > 1:
        final_words_list.add(word)
final_words_list.add('UNK')

# generate indexes for all the words and labels
word_indexes = {}
idx = 0
for word in final_words_list:
    word_indexes[word] = idx
    idx = idx + 1

# define the class labels and their inverted index
class_indexes = {}
classes_list = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
class_indexes = {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, \
                'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6, 'I-PER': 7, \
                'O': 8}

num_embeddings = len(word_indexes)
num_classes = len(class_indexes)

#generate encoded version of the data
for sentence in X_train_sentences:
    idxs = []
    for word in sentence:
        if word in word_indexes:
            idxs.append(word_indexes[word])
        else:
            idxs.append(word_indexes['UNK'])
    X_train_idxs.append(idxs)
for labels in Y_train_labels:
    idxs = []
    for label in labels:
        idxs.append(class_indexes[label])
    Y_train_idxs.append(idxs)

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    l :: array containing the word indexes
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)
    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]
    assert len(out) == len(l)
    return out



