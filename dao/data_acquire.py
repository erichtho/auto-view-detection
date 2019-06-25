#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import jieba
import pandas as pd
import re
from sqlalchemy import create_engine

__author__ = "charlene"
__time__ = "2019-05-15"

def get_news_from_sql(num='all', need_token=True):
    def token(string):
        return ' '.join(re.findall('[\w|\d]+', string))

    host = 'cdb-q1mnsxjb.gz.tencentcdb.com'
    user = 'root'
    password = 'A1@2019@me'
    database = 'news_chinese'
    port = '10102'
    con_engine = create_engine(
        'mysql://' + user + ':' + password + '@' + host + ':' + port + '/' + database + '?charset=utf8', encoding='utf-8')

    get_corpus_sql = """
        select content from sqlResult_1558435 {}
    """.format('limit '+str(num) if num!='all' and type(num)==int else '')
    raw_corpus = pd.read_sql(get_corpus_sql, con=con_engine)['content'].tolist()

    token_wrap = token if need_token else lambda t: t
    return [token_wrap(a) for a in raw_corpus]


def get_wiki_from_file():
    def token_wiki(string, del_pattern):
        return ' '.join(re.findall('[\w|\d]+', del_pattern.sub('', string)))

    wiki_articles = []
    wiki_path = "../data/wiki_expand_corpus/"
    for sub_folder in os.listdir(wiki_path):
        if sub_folder[0] != '.':
            for fname in os.listdir(wiki_path + sub_folder)[:10]:
                filepath = wiki_path + sub_folder + '/' + fname
                wiki_articles.append(open(filepath).read())

    tag_pattern = re.compile('<.*?>')


    return [token_wiki(article, tag_pattern) for article in wiki_articles]



def cut(string):
    return [word for word in jieba.cut(string) if word != '\n' and word.strip()]


def get_corpus():
    return get_news_from_sql() + get_wiki_from_file()


def get_cutted_corpus():
    corpus = get_corpus()

    def cutted_corpus_generator():
        for tokenized_article in corpus:
            yield cut(tokenized_article)

    return cutted_corpus_generator

