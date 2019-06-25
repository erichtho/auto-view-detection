#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import sys
import time

from bottle import Bottle, BaseTemplate, TEMPLATE_PATH, view, request, static_file
from stanfordcorenlp import StanfordCoreNLP

from dao.data_acquire import get_corpus
from definitions import WEB_DIR
from service.get_view import get_views, get_stanford_nlp
from service.train_saying_word import read_say_words

__author__ = "charlene"
__time__ = "2019-05-21"


def build_application(env='dev'):
    app = Bottle()

    # with app:
    # Our application object is now the default
    # for all shortcut functions and decorators

    nlp = get_stanford_nlp()

    say_words = read_say_words()

    with open("web/config.js", "w") as f:
        host = host_dict[env]
        f.write('getViewUrl = {{\n\turl:\'http://{}:{}\'\n}}'.format(host[0], host[1]))

    BaseTemplate.defaults['app'] = app  # XXX Template global variable
    TEMPLATE_PATH.insert(0, 'views')  # XXX Location of HTML templates

    #  XXX Routes to static content
    # @app.route('/<path:re:favicon.ico>')
    # @app.route('/static/<path:path>')
    # def static(path):
    #     'Serve static content.'
    #     return static_file(path, root='static/')

    #  XXX Index page
    @app.route('/')  # XXX URL to page
    @view('index')  # XXX Name of template
    def index():
        """
        :return:
        """
        return static_file('index.html', root=WEB_DIR)

    @app.route('/static/<filename:path>')
    def send_static(filename):
        return static_file(filename, root=WEB_DIR)

    @app.post('/views', name='view')
    def fetch_views():
        sentence = request.forms.getunicode('sentence')

        views = get_views(sentence, nlp, say_words)
        view_dict = [{"person": person, "verb": verb, "view": view} for person, verb, view in views]
        return json.dumps(view_dict)

    return app


host_dict = {
    'dev': ('localhost', 8000, True),
    'product': ('39.98.75.29', 9999, False)
}


def main():

    logging.basicConfig(
        filename='log/auto-view-detection_{}.log'.format(time.strftime("%Y%m%dT%H%M%S", time.localtime(time.time()))),
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO)
    logging.info('Started')
    # print command line arguments
    env = 'dev' if len(sys.argv) < 2 else sys.argv[1]
    logging.info('auto-view-detection app in {} environment started.'.format(env))

    _host, _port, _debug = host_dict[env]
    app = build_application(env)
    app.run(host=_host, port=_port, debug=_debug)
    logging.info('Finished')


if __name__ == "__main__":
    main()