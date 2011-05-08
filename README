Stanford NER - Jan 2009
----------------------------------------------

This package provides a high-performance machine learning based named
entity recognition system, including facilities to train models from
supervised training data and pre-trained models for English.

(c) 2002-2009.  The Board of Trustees of The Leland
    Stanford Junior University. All Rights Reserved. 

Original CRF code by Jenny Finkel.
Additional modules, features, internationalization, compaction, and
support code by Christopher Manning, Christopher Cox, Huy Nguyen and
Shipra Dingare, and Anna Rafferty.
This release prepared by Christopher Manning.

LICENSE 

The software is licensed under the full GPL.  Please see the file LICENCE.txt

For more information, bug reports, and fixes, contact:
    Christopher Manning
    Dept of Computer Science, Gates 1A
    Stanford CA 94305-9010
    USA
    java-nlp-support@lists.stanford.edu
    http://www-nlp.stanford.edu/software/CRF-NER.shtml

CONTACT

For questions about this distribution, please contact Stanford's JavaNLP group
at java-nlp-support@lists.stanford.edu.  We provide assistance on a best-effort
basis.

TUTORIAL

Quickstart guidelines, primarily for end users who wish to use the included NER
models, are below.  For further instructions on training your own NER model,
go to http://www-nlp.stanford.edu/software/crf-faq.shtml.

INCLUDED SERIALIZED MODELS / TRAINING DATA

The basic included serialized model is a 3 class NER tagger that can
label: PERSON, ORGANIZATION, and LOCATION entities.  It is included as
ner-eng-ie.crf-3-all2008.ser.gz and within the jar file.  It is trained
on data from CoNLL, MUC6, MUC7, and ACE.  Because this model is trained
on both US and UK newswire, it is fairly robust across the two domains.

We have also included a 4 class NER tagger trained on the CoNLL 2003
Shared Task training data that labels for PERSON, ORGANIZATION,
LOCATION, and MISC.  It is named ner-eng-ie.crf-4-conll.ser.gz .

All of the serialized classifiers come in two versions, the second of
which uses a distributional similarity lexicon to improve performance
(by about 1.5% F-measure).  These classifiers have additional features
which make them perform substantially better, but they require rather
more memory.  


QUICKSTART INSTRUCTIONS

This NER system requires Java 1.5 or later.   We have only tested it on
the SUN JVM.

Providing java is on your PATH, you should just be able to run an NER
GUI demonstration by just clicking.  It might work to double-click on
the stanford-ner.jar archive but this may well fail as the operating
system does not give Java enough memory for our NER system, so it is
safer to instead double click on the ner-gui.bat icon (Windows) or
ner-gui.sh (Linux/Unix/MacOSX).  Then, from the Classifier menu, either
load a CRF classifier from the classifiers directory of the distribution
or you should be able to use the Load Default CRF option.  You can then
either load a text file or web page from the File menu, or decide to use
the default text in the window.  Finally, you can now named entity tag
the text by pressing the Run NER button.

From a command line, you need to have java on your PATH and the
stanford-ner.jar file in your CLASSPATH.  (The way of doing this depends on
your OS/shell.)  The supplied ner.bat and ner.sh should work to allow
you to tag a single file.  For example, for Windows:

    ner file

Or on Unix/Linux you should be able to parse the test file in the distribution
directory with the command:

java -mx600m edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier classifiers/ner-eng.8class.better.crf.gz -textFile sample.txt

When run from a jar file, you also have the option of using a serialized
classifier contained in the jar file.  A default serialized classifier 
(ner-eng-ie.crf-3-all2008.ser.gz) is in the jar file and can be used by
just saying: 

java -mx300m -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -textFile sample.txt

If you use the -jar command, or double-click the jar file, NERGUI is
automatically started, and you will also be given the option (under the
'Classifier' menu item) to load a default supplied classifier:

java -mx300m -jar stanford-ner.jar


PROGRAMMATIC USE

The NERDemo file illustrates a couple of ways of calling the system
programatically.  You should get the same results from

java -mx300m NERDemo classifiers/ner-eng-ie.crf-3-all2008.ser.gz sample.txt

as from using CRFClassifier.  For more information on API calls, look in
the enclosed javadoc directory: load index.html in a browser and look
first at the edu.stanford.nlp.ie.crf package and CRFClassifier class.
If you wish to train your own NER systems, look also at the
edu.stanford.nlp.ie package NERFeatureFactory class. 


SERVER VERSION

The NER code may also be run as a server listening on a socket:

java -mx400m -cp stanford-ner.jar edu.stanford.nlp.ie.NERServer 1234

You can specify which model to load with flags, either one on disk:

java -mx400m -cp stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadClassifier classifiers/ner-eng-ie.crf-3-all2008.ser.gz 1234

Or if you have put a model inside the jar file:

java -mx400m -cp stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadJarClassifier ner-eng-ie.crf-3-all2008.ser.gz 1234


RUNNING CLASSIFIERS FROM INSIDE A JAR FILE

The software can run any serialized classifier from within a jar file by
giving the flag -loadJarClassifier resourceName .  An end user can make
their own jar files with the desired NER models contained inside.  The
serialized classifier must be located immediately under classifiers/ in
the jar file, with the name given.  This allows single jar file
deployment.


PERFORMANCE GUIDELINES

Performance depends on many factors.  Speed and memory use depend on
hardware, operating system, and JVM.  Accuracy depends on the data
tested on.  Nevertheless, in the belief that something is better than
nothing, here are some statistics from one machine on one test set, in
semi-realistic conditions (where the test data is somewhat varied).

ner-eng-ie.crf-3-all2006.ser.gz (older version of ner-eng-ie.crf-3-all2008.ser.gz)
Memory: 100 MB (on a 32 bit machine)
PERSON	ORGANIZATION	LOCATION
89.19	80.15		85.48

ner-eng-ie.crf-3-all2006-distsim.ser.gz (older version of ner-eng-ie.crf-3-all2008-distsim.ser.gz)
Memory: 320MB (on a 32 bit machine)
PERSON	ORGANIZATION	LOCATION
91.88	82.91		88.21
