import torch
import torch.nn.functional as tf
import torch.optim as optim
import torch.nn as nn
from collections import Counter
from pprint import pprint
import numpy as np

device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

with open('../data/ld1_sentences.txt', 'r') as fh:
    text = fh.read().strip().split('\n')

text = [x.strip().split(' ') for x in text]

max_sentence_length = max([len(x) for x in text])
print("LONGEST SENTENCE:", max_sentence_length)

all_text = [x for sublist in text for x in sublist]

with open('../data/ld1_labels.txt', 'r') as fh:
    labels = [x.strip() for x in fh.read().strip().split('\n')]

label_rdict = {x:i for i, x in enumerate(labels)}

from collections import Counter
ct = Counter(all_text)

# from pprint import pprint
# pprint(ct.most_common(1000))
VOCAB_SIZE = len(ct)
NUM_CLASSES = len(set(labels))
NUM_LAYERS = 1

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


label_vectors = []
for i, x in enumerate(labels):
    # label_vectors.append(vectorize_label(x, vocab_rdict))
    label_vectors.append([label_rdict[x]])

label_vectors = torch.tensor(label_vectors, device = device)

print("Vocab size:", VOCAB_SIZE)
print("Number of classes:", NUM_CLASSES)

train_size = len(sentence_vectors)
print(len(sentence_vectors), len(label_vectors))

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


model = WikiClassifier(50, 150, NUM_CLASSES).cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)

print(model)
for epoch in range(10):
    for i in range(train_size):
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence, label = sentence_vectors[i], label_vectors[i]
        output = model(sentence.long())
        print("MODELED")
        loss = loss_function(output[-1].unsqueeze(0), label)
        loss.backward()
        optimizer.step()
    print("COMPLETION")
    print(loss.item())
