#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

__author__ = "charlene"
__time__ = "2019-06-10"


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

W2V_PATH = os.path.join(ROOT_DIR, 'data/models/news-corpus-vectors-20190515T101810.kv')

STANFORD_NLP_PATH = os.path.join(ROOT_DIR, 'data/models/stanford-corenlp-full-2018-10-05/')

WEB_DIR = os.path.join(ROOT_DIR, 'web')