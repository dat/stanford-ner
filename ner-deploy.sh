#!/bin/sh

#This serves as documentation for all other non-servlet deployment options.
#Author: dth
#Date: March 2011

#settings
MEMORY=700m
STANFORD_NER=stanford-ner.jar
CLASSIFIER=classifiers/ner-eng-ie.crf-3-all2008-distsim.ser.gz

ENCODING='utf-8'	#(default: utf-8)
PORT=1234			#(default: 1234)
OUTPUT_FORMAT=slashTags # slashTags|xml|inlineXML (default: slashTags)
PRESERVE_SPACING="true"	# true|false (default: true)


#run the server
java -Xmx$MEMORY -cp $STANFORD_NER edu.stanford.nlp.ie.NERServer \
	-loadClassifier $CLASSIFIER -port $PORT -encoding $ENCODING \
	-outputFormat $OUTPUT_FORMAT -preserveSpacing $PRESERVE_SPACING
	
exit 0


#other possible modes

#GUI mode
java -Xmx$MEMORY -cp $STANFORD_NER edu.stanford.nlp.ie.crf.NERGUI

#batch processing of text file
java -Xmx$MEMORY -cp $STANFORD_NER edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier $CLASSIFIER -textFile $1

#client mode to ping NER server and test that the server is running
HOST=localhost
java -Xmx$MEMORY -cp $STANFORD_NER edu.stanford.nlp.ie.NERServer \
	-client true -host $HOST -port $PORT -encoding $ENCODING
