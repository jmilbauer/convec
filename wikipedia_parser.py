import xml.etree.ElementTree as etree
import codecs
import csv
import time
import os

DATA_PATH = './data/'
WIKIPEDIA_XML = 'enwiki-latest-pages-articles.xml'
ARTICLES_PATH = 'articles.csv'
REDIRECT_PATH = 'articles_redirect.csv'
TEMPLATE_PATH = 'articles_template.csv'
ENCODING = "utf-8"

def strip_tag_namespace(elem):
    t = elem.tag
    idx = k = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t

wikipediaXmlPath = os.path.join(DATA_PATH, WIKIPEDIA_XML)
articlesPath = os.path.join(DATA_PATH, articlesPath_PATH)
redirectPath = os.path.join(DATA_PATH, REDIRECT_PATH)
templatePath = os.path.join(DATA_PATH, TEMPLATE_PATH)

totalCount = 0
articleCount = 0
redirectCount = 0
templateCount = 0
title = None
start_time = time.time()
