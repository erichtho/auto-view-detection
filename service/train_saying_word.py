#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

import os
from gensim.models import KeyedVectors

from definitions import W2V_PATH

__author__ = "charlene"
__time__ = "2019-05-09"




# 3. word2Vec
word_vectors = KeyedVectors.load(W2V_PATH)
# 4. search for "说"
import itertools
from functools import lru_cache, wraps
from collections import defaultdict, OrderedDict


def concat_unique(lists):
    """
    合并一组list，并去掉其中的重复项
    """
    return list(set(itertools.chain(*lists)))




class RelatedWordSearch():
    def __init__(self, wv, thr=0.6, window=10):
        self.wv = wv
        self.thr = thr
        self.window = window

    @lru_cache(maxsize=10000)
    def get_expand(self, word):
        s = self.wv.similar_by_word(word, topn=self.window)
        return [(w, f) for w, f in s if f > self.thr]

    def search_bfs(self, initial_words, max_size):
        unseen = OrderedDict(zip(initial_words, [1] * len(initial_words)))

        seen = defaultdict(lambda: 1)

        while unseen and len(seen) < max_size:

            node_v, prob = unseen.popitem(last=False)
            # print('node: {}'.format(node))

            if node_v not in seen:
                new_expanding = self.get_expand(node_v)

                for w, f in new_expanding:
                    if w not in seen:
                        unseen[w] = f * prob

                unseen = OrderedDict(sorted(unseen.items(), key=lambda i: i[1], reverse=True))

                seen[node_v] *= prob

                if len(seen) % 50 == 0:
                    print("We have found {} related words.".format(len(seen)))

        return seen

word_cache = {}
def search_related_words(wv, max_size, thr=0.6, window=10):
    """
    获得word的所有同义词
    """

    @wraps(wv, max_size, thr, window)
    @lru_cache(maxsize=max_size)
    def search(word):
        #         print('We are expanding {}th layer.'.format(current_layer-1))
        word_cache[word] = 1
        print('Current word_cache len: {}'.format(len(word_cache)))
        print('Current word_cache keys: {}'.format(word_cache))

        if len(word_cache) >= max_size:
            return [word]
        else:
            print('wv.similar_by_word: {}'.format(wv.similar_by_word(word, topn=window)))
            return concat_unique(
                [[word]] +
                [search(w)
                 for w, s in wv.similar_by_word(word, topn=window)
                 if s >= thr])

    return search


def test():
    start_words = ['说', '提到', '表示', '提出']
    related_word = RelatedWordSearch(word_vectors, thr=0.7)
    r = related_word.search_bfs(start_words, 300)
    print(r)


def write_say_words(path='data/say_words.txt'):
    start_words = ['说', '提到', '表示', '提出']
    related_word = RelatedWordSearch(word_vectors, thr=0.7)
    r = related_word.search_bfs(start_words, 300)

    abspath = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), path)

    with open(abspath, 'w+') as f:
        f.write('\n'.join(['{} {}'.format(w, f) for w, f in r.items()]))


def read_say_words(path='data/say_words.txt', thr=0.35):

    abspath = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")), path)
    with open(abspath, 'r') as f:
        word_str = f.read()
        words = []
        for wf in word_str.split('\n'):
            w, f = tuple(wf.split())
            if float(f) > thr:
                words.append(w)

    return words

def main():
    write_say_words()


if __name__ == '__main__':
    main()