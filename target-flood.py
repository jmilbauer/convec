import xml.etree.ElementTree as etree
import codecs
import csv
import time
import os
import sys
import re
import cPickle as pickle
import nltk


DATA_PATH = './data/'
WIKIPEDIA_XML = sys.argv[1]

link_pattern = r'(\[\[([0-9A-z \(\)\.\-]*?)(?:\|([0-9A-z \(\)\.\-]*?))?\]\])'

wikipediaXmlPath = os.path.join(DATA_PATH, WIKIPEDIA_XML)


"""
specific doc grabber.
grabs the article titles for a flood distance of 2 for "Mathematics"


Usage example:

$ python target-flood.py 'testwiki.txt' 'Ayn Rand' 2
"""

SOURCE = sys.argv[2]
FLOOD_DEPTH = int(sys.argv[3])
tiers = {}
tiers[0] = set([SOURCE])
all_pages = set([SOURCE])

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

def surface_match(matchobj):
    if matchobj.group(3) == None:
        return matchobj.group(2)
    else:
        return matchobj.group(3)

def surface_form(sentence):
    return re.sub(link_pattern, surface_match, sentence)


for i in range(1, FLOOD_DEPTH+1):
    tiers[i] = set([])

    in_page = False
    ns = None
    going_to_read = False
    waiting = False
    for event, elem in etree.iterparse(wikipediaXmlPath, events=('start', 'end')):

        tname = strip_tag_namespace(elem.tag)

        if event == 'start':
            if tname == 'page':
                in_page = True

            if in_page:
                if tname == 'title':
                    if elem.text == None:
                        continue
                    else:
                        title = init_cap(elem.text)
                        if title in tiers[i-1]:
                            going_to_read = True

                if tname == 'redirect':
                    is_redirect = True
                if tname == 'ns':
                    ns = elem.text

                if tname == 'text' and ns == '0' and not is_redirect and going_to_read:
                    if elem.text == None:
                        waiting = True
                    else:
                        links = map(init_cap, return_links(elem.text))
                        for link in links:
                            if link not in all_pages:
                                tiers[i].add(link)
                            all_pages.add(link)
                        going_to_read = False

        if event == 'end':
            if in_page:
                if tname == 'text':
                    if waiting and ns == '0' and waiting and not is_redirect and going_to_read:
                        waiting = False
                        links = map(init_cap, return_links(elem.text))
                        for link in links:
                            if link not in all_pages:
                                tiers[i].add(link)
                            all_pages.add(link)
                        going_to_read = False

            if tname == 'page':
                in_page = False
                ns = None
                title = ''
                is_redirect = False
                going_to_read = False

        elem.clear()


for t in tiers:
    for page in list(sorted(tiers[t])):
        print("{}\t{}".format(page, t))
