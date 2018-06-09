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
TEXT_PATH = './text/'
WIKIPEDIA_XML = sys.argv[1]
TIER_PATH = 'distances.pickle'
XML_OUT = 'partial_xml.txt'
PREAMBLE = 'preamble.txt'
POSTAMBLE = 'postamble.txt'

link_pattern = r'(\[\[([0-9A-z \(\)\.\-]*?)(?:\|([0-9A-z \(\)\.\-]*?))?\]\])'

wikipediaXmlPath = os.path.join(DATA_PATH, WIKIPEDIA_XML)
tiersPath = os.path.join(DATA_PATH, TIER_PATH)
xmlPath = os.path.join(TEXT_PATH, XML_OUT)
prePath = os.path.join(TEXT_PATH, PREAMBLE)
postPath = os.path.join(TEXT_PATH, POSTAMBLE)

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


all_xml = []
for i in range(1, FLOOD_DEPTH+1):

    tiers[i] = set([])
    bodycount = 0
    partial_bodycount = 0
    in_page = False
    ns = None
    going_to_read = False
    waiting = False
    for event, elem in etree.iterparse(wikipediaXmlPath, events=('start', 'end')):

        tname = strip_tag_namespace(elem.tag)

        if event == 'start':
            if tname == 'page':
                in_page = True
                bodycount += 1
                partial_bodycount += 1
                if partial_bodycount == 1000:
                    partial_bodycount = 0
                    print("Link Depth: {}. Articles read: {}. Dataset size so far: {}".format(i, bodycount, len(all_pages)))
                xml_buffer = etree.tostring(elem)
                #xml_buffer.append('<page>')

            #else:

                #xml_buffer.append

            if in_page:
                if tname == 'title':
                    if elem.text == None or len(elem.text) < 1:
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
                        links = set([init_cap(l) for l in return_links(elem.text) if l not in all_pages])
                        tiers[i].union(links)
                        all_pages.union(links)
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

xml_final = []
xml_buffer = ""
in_page = False
ns = None
waiting = False
title = None
for event, elem in etree.iterparse(wikipediaXmlPath, events=('start', 'end')):

    tname = strip_tag_namespace(elem.tag)

    if event == 'start':
        if tname == 'page':
            in_page = True
        if in_page:
            if tname == 'title':
                if elem.text == None or len(elem.text) < 1:
                    continue
                else:
                    title = init_cap(elem.text)

            if tname == 'redirect':
                is_redirect = True

            if tname == 'ns':
                ns = elem.text

            if tname == 'text' and ns == '0' and not is_redirect and (title in all_pages):
                if elem.text == None:
                    waiting = True
                else:
                    fake = u"<page>"
                    fake += u"<title>{}</title>".format(title)
                    fake += u"<ns>0</ns>"
                    fake += u"<id>0</id>"
                    fake += u"<text>{}</text>".format(elem.text)
                    fake += u"</page>"
                    all_xml.append(fake)

    if event == 'end':
        if in_page:
            if tname == 'text':
                if waiting and ns == '0' and waiting and not is_redirect and (title in all_pages):
                    waiting = False
                    fake = u"<page>"
                    fake += u"<title>{}</title>".format(title)
                    fake += u"<ns>0</ns>"
                    fake += u"<id>0</id>"
                    fake += u"<text>{}</text>".format(elem.text)
                    fake += u"</page>"
                    all_xml.append(fake)

        if tname == 'page':
            in_page = False
            ns = None
            title = ''
            is_redirect = False

    elem.clear()




with open(xmlPath, 'wb+') as fp:
    pre = None
    post = None
    with open(prePath, 'rb') as fp2:
        pre = fp2.read()
    with open(postPath, 'rb') as fp3:
        post = fp3.read()

    fp.write(pre)
    for xml in all_xml:
        fp.write(xml.encode('utf-8'))
        fp.write('\n')
    fp.write(post)

with open(tiersPath, 'wb+') as fp:
    fp.write(pickle.dumps(tiers))
