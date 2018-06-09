import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import os
import sys
from nltk import word_tokenize
from nltk.corpus import stopwords
import time
import random

DATA_PATH = './../data/'
SENTENCES = 'ld1_sentences.txt'
LABELS = 'ld1_labels.txt'

sentencePath = os.path.join(DATA_PATH, SENTENCES)
labelPath = os.path.join(DATA_PATH, LABELS)

stopwords = set(stopwords.words('english'))

def tokenize():
    """
    returns context words for each label
    """

    sentences = []
    with open(sentencePath, 'rb') as fp:
        for line in fp:
            sentences.append(line.strip().split())

    labels = []
    with open(labelPath, 'rb') as fp:
        for line in fp:
            labels.append(line.strip())

    return zip(sentences, labels)

data = tokenize()
random.shuffle(data)
train_data = data[:1000]
holdout_data = data[1001:][:100]
test_data = data[1102:][:100]
labelset = set([])
vocabulary = set([])
for sentence, label in data:
    for token in sentence:
        if token not in stopwords:
            vocabulary.add(token)
    labelset.add(label)

word2id = {w : idx for (idx, w) in enumerate(vocabulary)}
id2word = {idx : w for (idx, w) in enumerate(vocabulary)}

label2id = {w : idx for (idx, w) in enumerate(labelset)}
id2label = {w : idx for (idx, w) in enumerate(labelset)}

vocab_size = len(vocabulary)
labelset_size = len(labelset)

data_pairs = []
for sentence, label in train_data:
    for token in sentence:
        if token not in stopwords:
            data_pairs.append((label2id[label], word2id[token]))
data_pairs = np.array(data_pairs)

holdout_pairs = []
for sentence, label in holdout_data:
    for token in sentence:
        if token not in stopwords:
            holdout_pairs.append((label2id[label], word2id[token]))
holdout_pairs = np.array(holdout_pairs)

print("Dataset size: {}".format(len(data_pairs)))

def get_input_layer(label_idx):
    x = torch.zeros(labelset_size).float()
    x[label_idx] = 1.0
    return x

embedding_dims = 20
W1 = Variable(torch.randn(embedding_dims, labelset_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocab_size, embedding_dims).float(), requires_grad=True)
num_epochs = 100
learning_rate = 1e-3

data_count = 0
start = time.time()
for epo in range(num_epochs):
    loss_val = 0
    for data, target in data_pairs:
        data_count += 1
        if data_count % 1000 == 0:
            print('{} datapoints. Laptime: {}'.format(data_count, time.time() - start))
            start = time.time()
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)

        log_softmax = F.log_softmax(z2, dim=0)
        print(log_softmax.view(1,-1))

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 1 == 0:
        print('Loss at epoch # {}: {}'.format(epo, loss_val / float(len(data_pairs))))

        for data, target in holdout_pairs:
            x = Variable(get_input_layer(data)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            z1 = torch.matmul(W1, x)
            z2 = torch.matmul(W2, z1)
            log_softmax = F.log_softmax(z2, dim=0)

            if log_softmax == y_true:
                print('correct')
            else:
                print('{}, {}'.format(log_softmax, y_true))











print(word2id)
