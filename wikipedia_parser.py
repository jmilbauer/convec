import xml.etree.ElementTree as etree
import codecs
import csv
import time
import os
import sys

"""
This code is heavily based on:
https://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html
"""

DATA_PATH = './data/'
WIKIPEDIA_XML = 'testwiki.txt'
ARTICLES_PATH = 'articles.csv'
REDIRECT_PATH = 'articles_redirect.csv'
TEMPLATE_PATH = 'articles_template.csv'
ENCODING = "utf-8"

def strip_tag_namespace(t):
    # t = elem.tag
    idx = k = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t

# get all the paths
wikipediaXmlPath = os.path.join(DATA_PATH, WIKIPEDIA_XML)
articlesPath = os.path.join(DATA_PATH, ARTICLES_PATH)
redirectPath = os.path.join(DATA_PATH, REDIRECT_PATH)
templatePath = os.path.join(DATA_PATH, TEMPLATE_PATH)

# initialize run observing data
totalCount = 0
articleCount = 0
redirectCount = 0
templateCount = 0
title = None
start_time = time.time()

# intialize reflective csvs
with codecs.open(articlesPath, 'w', ENCODING) as articlesFH, \
        codecs.open(redirectPath, 'w', ENCODING) as redirectFH, \
        codecs.open(templatePath, 'w', ENCODING) as templateFH:
    articlesWriter = csv.writer(articlesFH, quoting=csv.QUOTE_MINIMAL)
    redirectWriter = csv.writer(redirectFH, quoting=csv.QUOTE_MINIMAL)
    templateWriter = csv.writer(templateFH, quoting=csv.QUOTE_MINIMAL)

    articlesWriter.writerow(['id', 'title','redirect'])
    redirectWriter.writerow(['id', 'title','redirect'])
    templateWriter.writerow(['id', 'title'])

#
for event, elem in etree.iterparse(wikipediaXmlPath, events=('start', 'end')):
    tname = strip_tag_namespace(elem.tag)

    if event == 'start':
        if tname == 'page':
            title = ''
            i_d = -1
            redirect = ''
            inrevision = False
            ns = 0
        elif tname == 'revision':
            inrevision = True
    else:
        if tname == 'title':
            title = elem.text
        elif tname == 'id' and not inrevision:
            i_d = int(elem.text)
        elif tname == 'redirect':
            redirect = elem.attrib['title']
        elif tname == 'ns':
            print(elem)
            sys.exit()
            ns = int(elem.text)
        elif tname == 'page':
            totalCount += 1

            if ns == 10:
                templateCount += 1
                templateWriter.writerow([id, title])
            elif len(redirect) > 0:
                articleCount += 1
                articlesWriter.writerow([id, title, redirect])
            else:
                redirectCount += 1
                redirectWriter.writerow([id, title, redirect])

        if totalCount > 1 and (totalCount % 100000) == 0:
            print("{:,}".format(totalCount))

    elem.clear()

elapsed_time = time.time() - start_time

print("Total pages: {:,}".format(totalCount))
print("Template pages: {:,}".format(templateCount))
print("Article pages: {:,}".format(articleCount))
print("Redirect pages: {:,}".format(redirectCount))
print("Elapsed time: {}".format(elapsed_time))
