import time
import os
import sys
import re
import cPickle as pickle

"""
Computes a flood from a specific page.
Before using, make sure you have the adjacency matrix in your data folder.

$ python flood.py 'Source Article' [Flood Distance]

For example:

$ python flood.py 'Mathematics' 3

will pickle a dictionary with 4 entries:
res[0] = [Mathematics]
res[1] = pages 1 link from Mathematics
res[2] = pages 2 links from mathematics
res[3] = pages 3 links from Mathematics

"""


DATA_PATH = './data/'
ADJACENCY_FILE = 'adjacency.pickle'
REDIRECT_FILE = 'redirect.pickle'
ENCODING = "utf-8"
FLOOD_SOURCE = sys.argv[1]
FLOOD_FILE = '{}-flood.pickle'.format(FLOOD_SOURCE)
DISTANCE = sys.argv[2]

adjacencyPath = os.path.join(DATA_PATH, ADJACENCY_FILE)
redirectPath = os.path.join(DATA_PATH, REDIRECT_FILE)
floodPath = os.path.join(DATA_PATH, FLOOD_FILE)

adjacency_matrix = None
redirect_matrix = None

with open(adjacencyPath, 'rb') as fp:
    text = fp.read()
    adjacency_matrix = pickle.loads(text)

with open(redirectPath, 'rb') as fp:
    text = fp.read()
    redirect_matrix = pickle.loads(text)

tiers = {}
tiers[0] = set([FLOOD_SOURCE])

already_seen = set([FLOOD_SOURCE])
for i in range(1,DISTANCE):
    tiers[i] = set([])
    for page in tiers[i-1]:
        if page in adjacency_matrix:
            children = [k for k in adjacency_matrix[page].keys() if k not in already_seen]
            for child in children:
                already_seen.add(child)
                tiers[i].add(child)

with open(floodPath, 'wb+') as fp:
    fp.write(pickle.dumps(tiers))
