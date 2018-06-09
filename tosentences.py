import xml.etree.ElementTree as etree
import nltk
import os
import sys
import time
import re

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

DATA_PATH = './data/'
TEXT_PATH = './text/'
TEXT_FOLDER = 'extracted'
SENTENCES = 'sentences.txt'
LABELS = 'labels.txt'
ENCODING = "utf-8"


sentencePath = os.path.join(DATA_PATH, SENTENCES)
labelPath = os.path.join(DATA_PATH, LABELS)
extractedDir = os.path.join(TEXT_PATH, TEXT_FOLDER)

interior_pattern = r"(?s)<doc .*?title=\"(.*?)\">(.*?)<\/doc>"

data = []
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(extractedDir) for f in fn if not f.startswith('.')]

counter = 0
for filepath in files:

    text = None
    with open(filepath, 'rb') as fp:
        text = unicode(fp.read(), errors='ignore')

    for match in re.finditer(interior_pattern, text):
        body = match.group(2)
        title = match.group(1)
        sentences = tokenizer.tokenize(body)
        for sentence in sentences:
            data.append((sentence, title))


with open(sentencePath, 'wb+') as sentence_fp:
    with open(labelPath, 'wb+') as label_fp:
        for (s, l) in data:
            sentence_fp.write(s)
            sentence_fp.write('\n')

            label_fp.write(l)
            label_fp.write('\n')
