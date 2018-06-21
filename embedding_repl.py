import os
import pickle
import numpy as np
DATA_PATH = './data/'
EMBED_PATH = '2layer_50neuron_150emb_0.pkl'

embeddingPath = os.path.join(DATA_PATH, EMBED_PATH)

embeddings = None
with open(embeddingPath, 'rb') as fp:
    embeddings = pickle.load(fp)

print(list(embeddings.keys()))

def cosine_sim(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        print("Encountered zero-vector in sparse cosine similarity, stopping.")
        print("\tWordvec1: {}, Wordvec2: {}".format(vec1, vec2))
        sys.exit()
    return (float(np.dot(vec1, vec2))) / (norm1 * norm2)

def k_nearest_neighbors(k, word, m):
    """
    Finds the k nearest neighbors for a given word, based on m.
    TODO: use a heap.
    """
    if word in m:
        knn = []
        for w in m:
            if word == w:
                continue
            sim = cosine_sim(m[word], m[w])
            knn.append((sim, w))
            knn.sort(key = lambda x: x[0], reverse = True)
        knn = knn[:k]
        return [(w, round(x,4)) for (x, w) in knn]
    else:
        return []

while True:
    word = input('> ')
    if word in embeddings:
        print("Title found. Printing 10 NN.")
        knn = k_nearest_neighbors(10, word, embeddings)
        print(knn)
    else:
        print("Title not found.")
