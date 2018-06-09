import xml.etree.ElementTree as etree
import nltk
import os
import sys
import time
import re
import string

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

DATA_PATH = 'data/'
TEXT_PATH = 'text/'
TEXT_FOLDER = 'ld1_extracted'
SENTENCES = 'ld1_sentences.txt'
LABELS = 'ld1_labels.txt'
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

def clean_sentence(text):
    """Removes punctuation"""
    text = re.sub(r'([A-Za-z])\.([A-Za-z])', r'\1\2', text)
    text = re.sub(r'([A-Za-z ])\.\s*', r'\1 ', text.strip())
    text = re.sub(r'\.$', '', text)
    text = re.sub(r'[^\w\s\.\-]', r'', text)
    text = text.lower()
    text = re.sub(r'[0-9]+', 'NUM', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s*', ' ', text)
    text = text.strip()
    return text

with open(sentencePath, 'wb+') as sentence_fp:
    with open(labelPath, 'wb+') as label_fp:
        for (s, l) in data:

            cs = clean_sentence(s)

            if cs == '':
                continue
            else:
                sentence_fp.write(cs)
                sentence_fp.write('\n')

                label_fp.write(l.lower().replace(' ', '_'))
                label_fp.write('\n')
