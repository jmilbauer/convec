import torch
import torch.nn.functional as tf
import torch.optim as optim
import torch.nn as nn
from collections import Counter
from pprint import pprint
import numpy as np
import random
import time
import argparse
import pickle

parser = argparse.ArgumentParser(description = 'Train a LSTM classifier on wikipedia articles')
parser.add_argument('--num_layers', type = int, default = 1,
help = "Number of layers in the LSTM")
parser.add_argument('--hidden_layer_size', type = int, default = 150,
help = "Size of hidden layers (in neurons)")
parser.add_argument('--embedding_dim', type = int, default = 50,
help = "Dimensionality of word embedding layer")
parser.add_argument('--output_file', type = str, default = None,
help = "Output file for pickling embeddings")
parser.add_argument('--print_test_errors', action = 'store_true', default = False,
help = "Print errors during test set")
args = parser.parse_args()

NUM_LAYERS = args.num_layers
HIDDEN_LAYER_SIZE = args.hidden_layer_size
EMBEDDING_DIM = args.embedding_dim
OUTPUT_FILE = args.output_file
PRINT_TEST_ERRORS = args.print_test_errors

print("Network options:")
print("{} hidden layers".format(NUM_LAYERS))
print("{} neurons/hidden layer".format(HIDDEN_LAYER_SIZE))
print("{} embedding dim".format(EMBEDDING_DIM))

device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

with open('../data/ld1_sentences.txt', 'r') as fh:
    text = fh.read().strip().split('\n')

text = [x.strip().split(' ') for x in text]

# max_sentence_length = max([len(x) for x in text])
# print("LONGEST SENTENCE:", max_sentence_length)

all_text = [x for sublist in text for x in sublist]

with open('../data/ld1_labels.txt', 'r') as fh:
    labels = [x.strip() for x in fh.read().strip().split('\n')]

label_list = list(set(labels))
label_rdict = {x:i for i, x in enumerate(label_list)}

lengths = [11238, 1405, 1405] # lengths for train, dev, and test sets

indices = list(range(len(text)))
random.shuffle(indices)
train_indices, other = indices[:lengths[0]], indices[lengths[0]:]
dev_indices, test_indices = other[:lengths[1]], other[lengths[1]:]

test_sentences = [text[x] for x in test_indices]
real_test_labels = [labels[x] for x in test_indices]

# print(len(train_indices))
# print(len(dev_indices))
# print(len(test_indices))

from collections import Counter
ct = Counter(all_text)

# from pprint import pprint
# pprint(ct.most_common(1000))
VOCAB_SIZE = len(ct)
NUM_CLASSES = len(set(labels))

vocabulary = list(ct)
vocab_rdict = {x:i for i, x in enumerate(vocabulary)}

def vectorize_sentence(sentence, index_dict):
    x = np.zeros(len(sentence))
    for i, word in enumerate(sentence):
        x[i] = index_dict[word]
    return torch.tensor(x, device = device)

def vectorize_label(label, index_dict):
    x = np.zeros(NUM_CLASSES)
    x[label_rdict[label]] = 1
    return torch.tensor(x, device = device)

sentence_vectors = []
for i, x in enumerate(text):
    vec = torch.tensor(vectorize_sentence(x, vocab_rdict), device = device)
    # vec = vec.unsqueeze(0)
    sentence_vectors.append(vec)

train_vectors = [sentence_vectors[x] for x in train_indices]
dev_vectors = [sentence_vectors[x] for x in dev_indices]
test_vectors = [sentence_vectors[x] for x in test_indices]

label_vectors = []
for i, x in enumerate(labels):
    # label_vectors.append(vectorize_label(x, vocab_rdict))
    label_vectors.append([label_rdict[x]])

train_labels = torch.tensor([label_vectors[x] for x in train_indices], device = device)
dev_labels = torch.tensor([label_vectors[x] for x in dev_indices], device = device)
test_labels = torch.tensor([label_vectors[x] for x in test_indices], device = device)

label_vectors = torch.tensor(label_vectors, device = device)

print("Vocab size:", VOCAB_SIZE)
print("Number of classes:", NUM_CLASSES)

train_size = len(train_vectors)
dev_size = len(dev_vectors)
test_size = len(test_vectors)

"""Code for WikiClassifier class based in part on PyTorch's LSTMTagger example, from the documentation"""

class WikiClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers = 1):
        super(WikiClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.model = nn.LSTM(embedding_dim, hidden_dim, self.num_layers)
        self.hidden2class = nn.Linear(hidden_dim, output_dim)

        self.hidden_state = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_dim).cuda(), torch.zeros(self.num_layers, 1, self.hidden_dim).cuda())

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.model(embeds.view(len(sentence), 1, -1), self.hidden)
        class_space = self.hidden2class(lstm_out.view(len(sentence), -1))
        scores = tf.log_softmax(class_space, dim = 1)
        return scores


model = WikiClassifier(50, HIDDEN_LAYER_SIZE, NUM_CLASSES, num_layers = NUM_LAYERS).cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)

in_sample_loss = []
in_sample_accuracy = []
print(model)
prev_dev_loss = 500
epoch = 0
while True:
    isl = 0
    isa = 0
    start = time.time()
    for i in range(train_size):
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence, label = train_vectors[i], train_labels[i]
        output = model(sentence.long())
        loss = loss_function(output[-1].unsqueeze(0).cuda(), label)
        loss.backward()
        optimizer.step()
        values, indices = output[-1].max(0)
        if label.item() == indices.item():
            isa += 1
        isl += loss.item()
    end = time.time()
    print("Epoch {} complete".format(epoch))
    print("Time: {:.2f}".format(end-start))
    print("Average in-sample loss:", isl/train_size)
    print("In-sample accuracy:", isa/train_size)
    dsl = 0
    dsa = 0
    for i in range(dev_size):
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence, label = dev_vectors[i], dev_labels[i]
        output = model(sentence.long())
        loss = loss_function(output[-1].unsqueeze(0).cuda(), label)
        # loss.backward()
        # optimizer.step()
        values, indices = output[-1].max(0)
        if label.item() == indices.item():
            dsa += 1
        dsl += loss.item()
    print("Average dev loss:", dsl/dev_size)
    print("dev accuracy:", dsa/dev_size)
    if (dsl/dev_size > prev_dev_loss):
        print("Initiating early stopping")
        break
    prev_dev_loss = dsl/dev_size
    epoch += 1

tsl = 0
tsa = 0
for i in range(test_size):
    model.zero_grad()
    model.hidden = model.init_hidden()
    sentence, label = test_vectors[i], test_labels[i]
    output = model(sentence.long())
    loss = loss_function(output[-1].unsqueeze(0).cuda(), label)
    # loss.backward()
    # optimizer.step()
    values, indices = output[-1].max(0)
    if label.item() == indices.item():
        tsa += 1
    else:
        if PRINT_TEST_ERRORS:
            print("Sentence:", ' '.join(test_sentences[i]))
            print("Actual: {}".format(real_test_labels[i]))
            try:
                print("Predicted: {}".format(label_rdict[i]))
            except:
                pass
    tsl += loss.item()
print("Average test loss:", tsl/test_size)
print("Test accuracy:", tsa/test_size)

if OUTPUT_FILE is not None:
    weights = list(model.hidden2class.parameters())[0].detach().cpu().numpy()
    assert len(weights) == len(label_list)
    out_dict = {}
    for i in range(len(label_list)):
        out_dict[label_list[i]] = weights[i]
    with open(OUTPUT_FILE, 'wb') as fh:
        pickle.dump(out_dict, fh)
    print(out_dict)
    print("Successfully wrote to", OUTPUT_FILE)
