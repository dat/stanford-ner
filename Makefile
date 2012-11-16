
#Instructions:
# - Execute 'make war' to make a war file for Tomcat deployment.
# - Execute 'make jar' to make a jar library file.
# - Execute 'make clean' to clean up the working space.

JAVAC = javac
JAVAFLAGS = -O
JAVADOC = javadoc

SERVLET_API = extra/servlet-api.jar

SHELL = /bin/bash

all: war

init:
	mkdir -p tmp
	mkdir -p doc

doc: init
	$(JAVADOC) -d doc -sourcepath src -subpackages com edu

war: init
	mkdir -p tmp/META-INF tmp/WEB-INF
	mkdir -p tmp/WEB-INF/classes tmp/WEB-INF/lib tmp/WEB-INF/resources
	mkdir -p tmp/WEB-INF/resources/classifiers
	cp classifiers/*.ser.gz tmp/WEB-INF/resources/classifiers
	cp extra/index.html tmp
	cp extra/LICENSE.txt tmp/META-INF
	cp extra/web.xml tmp/WEB-INF
 
	$(JAVAC) $(JAVAFLAGS) -d tmp/WEB-INF/classes \
		-classpath $(SERVLET_API) \
		src/edu/stanford/nlp/*/*.java \
        src/edu/stanford/nlp/*/*/*.java \
		src/com/ntrepid/tartan/*.java
	pushd tmp && jar -cfm ../stanford-ner.war ../src/edu/stanford/nlp/ie/crf/ner-manifest.txt * && popd

jar: init
	mkdir -p tmp/META-INF
	cp extra/LICENSE.txt tmp/META-INF
	$(JAVAC) $(JAVAFLAGS) -d tmp \
		src/edu/stanford/nlp/*/*.java src/edu/stanford/nlp/*/*/*.java
	pushd tmp && jar -cfm ../stanford-ner.jar ../src/edu/stanford/nlp/ie/crf/ner-manifest.txt * && popd

clean:
	rm -fR tmp *.war *.jar
	rm -fR doc

.PHONY: all init doc war jar clean

