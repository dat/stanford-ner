#!/usr/bin/env python2.7

"""This module provides a Python wrapper for the Stanford NER
running over the raw socket or HTTP request."""


import socket
import re
import httplib
import urllib
from contextlib import contextmanager
from itertools import groupby
from operator import itemgetter
from stanfordner.exceptions import StanfordNERException, InvalidOutputFormat


@contextmanager
def tcpip4_socket(host, port):
    """Open a TCP/IP4 socket to designated host/port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        yield s
    finally:
        s.shutdown(socket.SHUT_RDWR)
        s.close()

@contextmanager
def http_connection(host, port):
    """Open an HTTP connection to designated host/port."""
    c = httplib.HTTPConnection(host, port)
    try:
       yield c
    finally:
       c.close()


#regex patterns of various tagging options (for parsing)
SLASHTAGS_EPATTERN = re.compile(r'(.+?)/([A-Z]+)\s*')
XML_EPATTERN = re.compile(r'<wi num=".+?" entity="(.+?)">(.+?)</wi>')
INLINEXML_EPATTERN = re.compile(r'<([A-Z]+?)>(.+?)</\1>')


class StanfordNER(object):
    """Wrapper for server-based Stanford NER tagger."""

    def tag_text(self, text):
        pass

    def __slashTags_parse_entities(self, tagged_text):
        """Return a list of token tuples (entity_type, token) parsed
        from slashTags-format tagged text."""
        return (match.groups()[::-1] for match in
            SLASHTAGS_EPATTERN.finditer(tagged_text))

    def __xml_parse_entities(self, tagged_text):
        """Return a list of token tuples (entity_type, token) parsed
        from xml-format tagged text."""
        return (match.groups() for match in
            XML_EPATTERN.finditer(tagged_text))

    def __inlineXML_parse_entities(self, tagged_text):
        """Return a list of entity tuples (entity_type, entity) parsed
        from inlineXML-format tagged text."""
        return (match.groups() for match in
            INLINEXML_EPATTERN.finditer(tagged_text))

    def __collapse_to_dict(self, pairs):
        """Return a dictionary mapping the first value of every pair
        to a collapsed list of all the second values of every pair."""
        return dict((first, map(itemgetter(1), second)) for (first, second)
            in groupby(sorted(pairs, key=itemgetter(0)), key=itemgetter(0)))

    def get_entities(self, text):
        """Extract all the named entities in text."""
        tagged_text = self.tag_text(text)
        if self.oformat == 'slashTags':
            entities = self.__slashTags_parse_entities(tagged_text)
            entities = ((etype, " ".join(t[1] for t in tokens)) for (etype, tokens) in
                groupby(entities, key=itemgetter(0)))
        elif self.oformat == 'xml':
            entities = self.__xml_parse_entities(tagged_text)
            entities = ((etype, " ".join(t[1] for t in tokens)) for (etype, tokens) in
                groupby(entities, key=itemgetter(0)))
        else: #inlineXML
            entities = self.__inlineXML_parse_entities(tagged_text)
        return self.__collapse_to_dict(entities)


class StanfordNERSocket(StanfordNER):
    """Stanford NER based on raw socket."""

    def __init__(self, host='localhost', port=1234, output_format='inlineXML'):
        if output_format not in ('slashTags', 'xml', 'inlineXML'):
            raise InvalidOutputFormat('Output format %s is invalid.' % output_format)
        self.host = host
        self.port = port
        self.oformat = output_format

    def tag_text(self, text):
        """Tag the text with proper named entities token-by-token."""
        for s in ('\f', '\n', '\r', '\t', '\v'): #strip whitespaces
            text = text.replace(s, '')
        text += '\n' #ensure end-of-line
        with tcpip4_socket(self.host, self.port) as s:
            s.sendall(text)
            tagged_text = s.recv(10*len(text))
        return tagged_text


class StanfordNERHTTP(StanfordNER):
    """Stanford NER based on HTTP request."""

    def __init__(self, host='localhost', port=1234, location='/stanford-ner/ner',
            classifier=None, output_format='inlineXML', preserve_spacing=True):
        if output_format not in ('slashTags', 'xml', 'inlineXML'):
            raise InvalidOutputFormat('Output format %s is invalid.' % output_format)
        self.host = host
        self.port = port
        self.location = location
        self.oformat = output_format
        self.classifier = classifier
        self.spacing = preserve_spacing

    def tag_text(self, text):
        """Tag the text with proper named entities token-by-token."""
        for s in ('\f', '\n', '\r', '\t', '\v'): #strip whitespaces
            text = text.replace(s, '')
        text += '\n' #ensure end-of-line
        with http_connection(self.host, self.port) as c:
            headers = {'Content-type': 'application/x-www-form-urlencoded', 'Accept' : 'text/plain'}
            if self.classifier:
                params = urllib.urlencode(
                    {'input': text, 'outputFormat': self.oformat, 
                    'preserveSpacing': self.spacing, 
                    'classifier': self.classifier})
            else:
                params = urllib.urlencode(
                    {'input': text, 'outputFormat': self.oformat, 
                    'preserveSpacing': self.spacing})
            try:
                c.request('POST', self.location, params, headers)
                response = c.getresponse()
                result = response.read()
            except httplib.HTTPException, e:
                print "Failed to post HTTP request."
                raise e
        return result
