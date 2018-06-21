import pickle
target = '2layer_50neuron_150emb_0.pkl'
with open(target, 'rb') as fh:
    d = pickle.load(fh)

labels = list(d.keys())
weights = list(d.values())

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys

tsne = TSNE(n_components = 2, random_state = 0)
Y = tsne.fit_transform(weights)

plt.scatter(Y[:, 0], Y[:, 1])
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext = (0, 0),
            textcoords='offset points')
plt.title('t-SNE projection of "Great American Novel" articles')
# plt.savefig('t-SNE.png', dpi = 500, bbox_inches = 'tight', pad_inches = 0.5)
plt.show()
