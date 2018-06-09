import xml.etree.ElementTree as etree
import codecs
import csv
import time
import os
import sys
import re
import cPickle as pickle

"""
Referenced https://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html to learn to use iterparse.
"""



DATA_PATH = './data/'
WIKIPEDIA_XML = sys.argv[1]
SENTENCE_TSV = 'sentences.txt'
ADJACENCY_FILE = 'adjacency.pickle'
REDIRECT_FILE = 'redirect.pickle'
ENCODING = "utf-8"

link_pattern = r'(\[\[([A-z ]*?)(?:\|([A-z ]*?))?\]\])'

# get all the paths
wikipediaXmlPath = os.path.join(DATA_PATH, WIKIPEDIA_XML)
sentencePath = os.path.join(DATA_PATH, SENTENCE_TSV)
adjacencyPath = os.path.join(DATA_PATH, ADJACENCY_FILE)
redirectPath = os.path.join(DATA_PATH, REDIRECT_FILE)

# adjacency matrix
adjacency_matrix = {}
redirect_lookup = {}

def init_cap(string):
    l = list(string)
    l[0] = l[0].upper()
    return ''.join(l)

def strip_tag_namespace(t):
    """
    This method is from: https://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html
    """
    # t = elem.tag
    idx = k = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t

def return_links(text):
    results = []
    for match in re.finditer(link_pattern, text):
        results.append(match.group(2))
    return results

def process_text(article, text):
    adjacency_matrix[article] = {}
    targets = return_links(text)
    #surface_text =

    for target in targets:
        target = target
        if target in redirect_lookup:
            t = redirect_lookup[target]
            adjacency_matrix[article][t] = 1
        else:
            adjacency_matrix[article][target] = 1

    # print('For article: {}'.format(article))
    # print(u'\t{}'.format(text))


def get_redirect_dictionary():
    in_page = False
    title = ""
    res = {}
    for event, elem in etree.iterparse(wikipediaXmlPath, events=('start', 'end')):
        tname = strip_tag_namespace(elem.tag)

        if event == 'start':
            if tname == 'page':
                in_page = True

            if in_page:

                if tname == 'title':
                    title = init_cap(elem.text)

                if tname == 'redirect' and in_page and title != "":
                    res[title] = init_cap(elem.attrib['title'])

        if event == 'end':
            if tname == 'page':
                in_page = False
                title = ""
    return res

def build_adjacency_matrix():
    in_page = False
    ns = None
    body_count = 0
    title = ''
    waiting = False
    is_redirect = False

    for event, elem in etree.iterparse(wikipediaXmlPath, events=('start', 'end')):
        tname = strip_tag_namespace(elem.tag)

        if event == 'start':
            if tname == 'page':
                in_page = True

            if in_page:

                if tname == 'title':
                    title = init_cap(elem.text)
                if tname == 'redirect':
                    is_redirect = True
                if tname == 'ns':
                    ns = elem.text

                if tname == 'text' and ns == '0' and not is_redirect:
                    if elem.text == None:
                        waiting = True
                    else:
                        process_text(title, elem.text)
                        body_count += 1

        if event == 'end':
            if in_page:
                if tname == 'text':
                    if waiting and ns == '0' and waiting and not is_redirect:
                        waiting = False
                        process_text(title, elem.text)
                        body_count += 1

            if tname == 'page':
                in_page = False
                ns = None
                title = ''
                is_redirect = False

        elem.clear()

    return body_count

start = time.time()
redirect_lookup = get_redirect_dictionary()
article_count = build_adjacency_matrix()
elapsed_time = time.time() - start

print("Time Elapsed: {}".format(elapsed_time))
print("Found text for {} articles".format(article_count))

start = time.time()
with open(adjacencyPath, 'wb+') as fp:
    fp.write(pickle.dumps(adjacency_matrix))

with open(redirectPath, 'wb+') as fp:
    fp.write(pickle.dumps(redirect_lookup))
elapsed_time = time.time() - start

print("Pickled. time: {}".format(elapsed_time))
