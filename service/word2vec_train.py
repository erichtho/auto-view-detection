#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import re

import jieba
import pandas as pd

from sqlalchemy import create_engine

from dao.data_acquire import get_cutted_corpus

__author__ = "charlene"
__time__ = "2019-05-09"


# 网上抄的sentence的wrapper，把generator包装成Word2Vec的sentences参数可接受的类型，
# gensim.models.Word2Vec的sentences不能是generator，
# 尽管他的官方教程说可以（https://radimrehurek.com/gensim/models/word2vec.html），
# 原因据说是Word2Vec需要在sentences上iterate两次
class SentenceIterator():
    def __init__(self, generate_func):
        self.generate_func = generate_func
        self.generator = self.generate_func()

    def __iter__(self):
        self.generator = self.generate_func()
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result




from gensim.models import Word2Vec

model = Word2Vec(SentenceIterator(get_cutted_corpus()), size=100, window=5, min_count=1, workers=4)

ver_no = time.strftime("%Y%m%dT%H%M%S", time.localtime(time.time()))

fname = "../data/models/news-corpus-vectors-{}.kv".format(ver_no)
model.wv.save(fname)

