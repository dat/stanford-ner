#!/usr/bin/env python2.7

"""Python wrapper for the Stanford NER.
@author Dat Hoang
@date March 2011"""


from stanfordner.client import StanfordNERSocket, StanfordNERHTTP
from stanfordner.exceptions import StanfordNERException, InvalidOutputFormat

__version__ = '0.1'

__all__ = [
    'StanfordNERSocket', 'StanfordNERHTTP',
    'StanfordNERException', 'InvalidOutputFormat']
