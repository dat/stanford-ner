// AbstractSequenceClassifier -- a framework for probabilistic sequence models.
// Copyright (c) 2002-2008 The Board of Trustees of
// The Leland Stanford Junior University. All Rights Reserved.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
// For more information, bug reports, fixes, contact:
//    Christopher Manning
//    Dept of Computer Science, Gates 1A
//    Stanford CA 94305-9010
//    USA
//    Support/Questions: java-nlp-user@lists.stanford.edu
//    Licensing: java-nlp-support@lists.stanford.edu
//    http://nlp.stanford.edu/downloads/crf-classifier.shtml

package edu.stanford.nlp.ie;

import edu.stanford.nlp.fsm.DFSA;
import edu.stanford.nlp.io.RegExFileFilter;
import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.objectbank.ObjectBank;
import edu.stanford.nlp.objectbank.ResettableReaderIteratorFactory;
import edu.stanford.nlp.sequences.*;
import edu.stanford.nlp.sequences.FeatureFactory;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.Sampler;
import edu.stanford.nlp.util.*;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

/** This class provides common functionality for (probabilistic) sequence
 *  models.  It is a superclass of our CMM and CRF sequence classifiers,
 *  and is even used in the (deterministic) NumberSequenceClassifier.
 *  See implementing classes for more information.
 *
 *  @author Jenny Finkel
 *  @author Dan Klein
 *  @author Christopher Manning
 *  @author Dan Cer
 */
public abstract class AbstractSequenceClassifier implements Function<String, String> {

  public static final String JAR_CLASSIFIER_PATH = "/classifiers/";

  public SeqClassifierFlags flags;
  public Index<String> classIndex;  // = null;
  protected DocumentReaderAndWriter readerAndWriter; // = null;
  public FeatureFactory featureFactory;
  protected CoreLabel pad;
  public int windowSize;
  protected Set<String> knownLCWords = new HashSet<String>();



  /** Construct a SeqClassifierFlags object based on the passed in properties,
   *  and then call the other constructor.
   *
   *  @param props See SeqClassifierFlags for known properties.
   */
  public AbstractSequenceClassifier(Properties props) {
    this(new SeqClassifierFlags(props));
  }


  /** Initialize the featureFactor and other variables based on the passed in
   *  flags.
   *
   *  @param flags A specification of the AbstractSequenceClassifier to construct.
   */
  public AbstractSequenceClassifier(SeqClassifierFlags flags) {
    this.flags = flags;
    pad = new CoreLabel();
    windowSize = flags.maxLeft + 1;
    try {
      featureFactory = (FeatureFactory) Class.forName(flags.featureFactory).newInstance();
    } catch (Exception e) {
      e.printStackTrace();
      throw new RuntimeException(e.getMessage());
    }
    reinit();
  }


  /** This method should be called after there have been changes to the
   *  flags (SeqClassifierFlags) variable, such as after deserializing
   *  a classifier.  It is called inside the loadClassifier methods.
   *  It assumes that the flags variable and the pad
   *  variable exist, but reinitializes things like the pad variable,
   *  featureFactory and readerAndWriter based on the flags.
   *  <p>
   *  <i>Implementation note:</i> At the moment this variable doesn't
   *  set windowSize or featureFactory, since they are being serialized
   *  separately in the
   *  file, but we should probably stop serializing them and just
   *  reinitialize them from the flags?
   */
  protected final void reinit() {
    pad.set(AnswerAnnotation.class, flags.backgroundSymbol);
    pad.set(GoldAnswerAnnotation.class, flags.backgroundSymbol);

    try {
      readerAndWriter = (DocumentReaderAndWriter) Class.forName(flags.readerAndWriter).newInstance();
    } catch (Exception e) {
      e.printStackTrace();
      throw new RuntimeException(e.getMessage(), e);
    }
    readerAndWriter.init(flags);
    featureFactory.init(flags);
  }


  public String backgroundSymbol() {
    return flags.backgroundSymbol;
  }

  public Set<String> labels() {
    return new HashSet<String>(classIndex.objectsList());
  }


  /**
   * Classify a {@link Sentence}.
   *
   * @param sentence The {@link Sentence} to be classified.
   * @return The classified {@link Sentence}, where the classifier output for
   * each token is stored in its "answer" field.
   */
  public List<CoreLabel> classifySentence(List<? extends HasWord> sentence) {
    List<CoreLabel> document = new ArrayList<CoreLabel>();
    int i = 0;
    for (HasWord word : sentence) {
      CoreLabel wi = new CoreLabel();
      wi.setWord(word.word());
      wi.set(PositionAnnotation.class, Integer.toString(i));
      wi.set(AnswerAnnotation.class, backgroundSymbol());
      document.add(wi);
      i++;
    }
    ObjectBankWrapper wrapper = new ObjectBankWrapper(flags, null, knownLCWords);
    wrapper.processDocument(document);

    classify(document);

    return document;
  }

  public SequenceModel getSequenceModel(List<? extends CoreLabel> doc) {
    throw new UnsupportedOperationException();
  }

  public Sampler<List<CoreLabel>> getSampler(final List<? extends CoreLabel> input) {
    return new Sampler<List<CoreLabel>>() {
      SequenceModel model = getSequenceModel(input);
      SequenceSampler sampler = new SequenceSampler();
      public List<CoreLabel> drawSample() {
        int[] sampleArray = sampler.bestSequence(model);
        List<CoreLabel> sample = new ArrayList<CoreLabel>();
        int i=0;
        for (CoreLabel word : input) {
          CoreLabel newWord = new CoreLabel(word);
          newWord.set(AnswerAnnotation.class, classIndex.get(sampleArray[i++]));
          sample.add(newWord);
        }
        return sample;
      }
    };
  }


  public Counter<List<CoreLabel>> classifyKBest(List<CoreLabel> doc, Class<? extends CoreAnnotation<String>> answerField, int k) {

    if (doc.isEmpty()) {
      return new ClassicCounter<List<CoreLabel>>();
    }

    // i'm sorry that this is so hideous - JRF
    ObjectBankWrapper obw = new ObjectBankWrapper(flags, null, knownLCWords);
    doc = obw.processDocument(doc);

    SequenceModel model = getSequenceModel(doc);

    KBestSequenceFinder tagInference = new KBestSequenceFinder();
    Counter<int[]> bestSequences = tagInference.kBestSequences(model,k);

    Counter<List<CoreLabel>> kBest = new ClassicCounter<List<CoreLabel>>();

    for (int[] seq : bestSequences.keySet()) {
      List<CoreLabel> kth = new ArrayList<CoreLabel>();
      int pos = model.leftWindow();
      for (CoreLabel fi : doc) {
        CoreLabel newFL = new CoreLabel(fi);
        String guess = classIndex.get(seq[pos]);
        fi.remove(AnswerAnnotation.class); // because fake answers will get added during testing
        newFL.set(answerField, guess);
        pos++;
        kth.add(newFL);
      }
      kBest.setCount(kth, bestSequences.getCount(seq));
    }

    return kBest;
  }


  @SuppressWarnings({"UnusedDeclaration"})
  public DFSA<String, Integer> getViterbiSearchGraph(List<CoreLabel> doc, Class<? extends CoreAnnotation<String>> answerField) {
    if (doc.isEmpty()) {
      return new DFSA<String, Integer>(null);
    }
    ObjectBankWrapper obw = new ObjectBankWrapper(flags, null, knownLCWords);
    doc = obw.processDocument(doc);
    SequenceModel model = getSequenceModel(doc);
    return ViterbiSearchGraphBuilder.getGraph(model, classIndex);
  }


  /**
   * Classify a List of CoreLabels using a TrueCasingDocumentReader.
   * <i>Note:</i> This was fairly quickly added to build a Truecaser.  It may
   * be revised or disappear.
   *
   * @param sentence a list of CoreLabels to be classifierd
   * @return The classified list}.
   */
  public List<CoreLabel> classifyWithCasing(List<CoreLabel> sentence) {
    List<CoreLabel> document = new ArrayList<CoreLabel>();
    int i = 0;
    for (CoreLabel word : sentence) {
      CoreLabel wi = new CoreLabel();
      if (readerAndWriter instanceof TrueCasingDocumentReaderAndWriter) {
        wi.setWord(word.word().toLowerCase());
        if (flags.useUnknown) {
          wi.set(UnknownAnnotation.class, (TrueCasingDocumentReaderAndWriter.known(wi.word()) ? "false" : "true"));
          //System.err.println(wi.word()+" : "+wi.get("unknown"));
        }
      } else {
        wi.setWord(word.word());
      }
      wi.set(PositionAnnotation.class, Integer.toString(i));
      wi.set(AnswerAnnotation.class, backgroundSymbol());
      document.add(wi);
      i++;
    }
    classify(document);
    i = 0;
    for (CoreLabel wi : document) {
      CoreLabel word = sentence.get(i);
      if (flags.readerAndWriter.equalsIgnoreCase("edu.stanford.nlp.sequences.TrueCasingDocumentReader")) {
        String w = word.word();
        if (wi.get(AnswerAnnotation.class).equals("INIT_UPPER") || wi.get(PositionAnnotation.class).equals(flags.backgroundSymbol)) {
          w = w.substring(0,1).toUpperCase()+w.substring(1).toLowerCase();
        } else if (wi.get(AnswerAnnotation.class).equals("LOWER")) {
          w = w.toLowerCase();
        } else if (wi.get(AnswerAnnotation.class).equals("UPPER")) {
          w = w.toUpperCase();
        }
        word.setWord(w);
      } else {
        word.setNER(wi.get(AnswerAnnotation.class));
      }
      i++;
    }
    return sentence;
  }

  /**
   * Classify the tokens in a String.  Each sentence becomes a separate
   * document.
   *
   * @param str A String with tokens in one or more sentences of text
   *                  to be classified.
   * @return {@link List} of classified sentences (each a List of
   *                 {@link CoreLabel}s).
   */
  public List<List<CoreLabel>> classify(String str) {
    DocumentReaderAndWriter oldRW = readerAndWriter;
    readerAndWriter = new PlainTextDocumentReaderAndWriter();
    readerAndWriter.init(flags);
    ObjectBank<List<CoreLabel>> documents = makeObjectBankFromString(str);
    List<List<CoreLabel>> result = new ArrayList<List<CoreLabel>>();

    for (List<CoreLabel> document : documents) {
      classify(document);

      List<CoreLabel> sentence = new ArrayList<CoreLabel>();
      for (CoreLabel wi : document) {
        // TaggedWord word = new TaggedWord(wi.word(), wi.answer());
        // sentence.add(word);
        sentence.add(wi);
      }
      result.add(sentence);
    }
    readerAndWriter = oldRW;
    return result;
  }

  /**
   * Classify the contents of a file.
   *
   * @param filename Contains the sentence(s) to be classified.
   * @return {@link List} of classified {@link Sentence}s.
   */
  public List<List<CoreLabel>> classifyFile(String filename) {
    DocumentReaderAndWriter oldRW = readerAndWriter;
    readerAndWriter = new PlainTextDocumentReaderAndWriter();
    readerAndWriter.init(flags);
    ObjectBank<List<CoreLabel>> documents = makeObjectBankFromFile(filename);
    List<List<CoreLabel>> result = new ArrayList<List<CoreLabel>>();

    for (List<CoreLabel> document : documents) {
      // System.err.println(document);
      classify(document);

      List<CoreLabel> sentence = new ArrayList<CoreLabel>();
      for (CoreLabel wi : document) {
        sentence.add(wi);
        // System.err.println(wi);
      }
      result.add(sentence);
    }
    readerAndWriter = oldRW;
    return result;
  }


  /**
   * Maps a String input to an XML-formatted rendition of applying NER to
   * the String.  Implements the Function interface.  Calls
   * classifyWithInlineXML(String) [q.v.].
   */
  public String apply(String in) {
    return classifyWithInlineXML(in);
  }

  /**
   * Classify the contents of a {@link String}.  Plain text or XML input is
   * expected and the {@link PlainTextDocumentReaderAndWriter} is used.
   * The classifier will tokenize the text and treat each sentence as a
   * separate document.
   * The output can be specified to be in a choice of three formats: slashTags
   * (e.g., Bill/PERSON Smith/PERSON died/O ./O), inlineXML
   * (e.g., &lt;PERSON&gt;Bill Smith&lt;/PERSON&gt;
   * went to &lt;LOCATION&gt;Paris&lt;/LOCATION&gt; .), or xml, for stand-off
   * XML (e.g., &lt;wi num="0" entity="PERSON"&gt;Sue&lt;/wi&gt;
   * &lt;wi num="1" entity="O"&gt;shouted&lt;/wi&gt; ).
   * There is also a binary choice as to whether the spacing between tokens
   * of the original is preserved or whether the (tagged) tokens are printed
   * with a single space (for inlineXML or slashTags) or a single newline
   * (for xml) between each one.
   * <p>
   * <i>Fine points:</i>
   * The slashTags and xml formats show tokens as transformed
   * by any normalization processes inside the tokenizer, while inlineXML
   * shows the tokens exactly as they appeared in the source text.
   * When a period counts as both part of an abbreviation and as an end of
   * sentence marker, it is included twice in the output String for slashTags
   * or xml, but only once for inlineXML, where it is not counted as part of
   * the abbreviation (or any named entity it is part of).  For slashTags with
   * preserveSpacing=true, there will be two successive periods such as "Jr.."
   * The tokenized (preserveSpacing=false) output will have a space or a
   * newline after the last token.
   *
   * @param sentences The String to be classified. It will be tokenized and
   *     divided into documents according to (heuristically determined)
   *     sentence boundaries.
   * @param outputFormat The format to put the output in: one of "slashTags",
   *     "xml", or "inlineXML"
   * @param preserveSpacing Whether to preserve the input spacing between
   *     tokens, which may sometimes be none (true) or whether to tokenize
   *     the text and print it with one space between each token (false)
   * @return A {@link String} with annotated with classification
   *         information.
   */
  public String classifyToString(String sentences,
                                 String outputFormat,
                                 boolean preserveSpacing) {
    int outFormat = PlainTextDocumentReaderAndWriter.asIntOutputFormat(outputFormat);

    DocumentReaderAndWriter tmp = readerAndWriter;
    readerAndWriter = new PlainTextDocumentReaderAndWriter();
    readerAndWriter.init(flags);

    ObjectBank<List<CoreLabel>> documents = makeObjectBankFromString(sentences);

    StringBuilder sb = new StringBuilder();
    for (List<CoreLabel> doc : documents) {
      classify(doc);
      sb.append(((PlainTextDocumentReaderAndWriter) readerAndWriter).getAnswers(doc, outFormat, preserveSpacing));
    }
    readerAndWriter = tmp;
    return sb.toString();
  }


  /**
   * Classify the contents of a {@link String}.  Plain text or XML is
   * expected and the {@link PlainTextDocumentReaderAndWriter} is used.
   * The classifier will treat each sentence as a separate document.
   * The output can be specified to be in a choice of formats:
   * Output
   * is in inline XML format (e.g. &lt;PERSON&gt;Bill Smith&lt;/PERSON&gt;
   * went to &lt;LOCATION&gt;Paris&lt;/LOCATION&gt; .)
   *
   * @param sentences The string to be classified
   * @return A {@link String} with annotated with classification
   *         information.
   */
  public String classifyWithInlineXML(String sentences) {
    return classifyToString(sentences, "inlineXML", true);
  }


  /**
   * Classify the contents of a String to a tagged word/class String.
   * Plain text or XML input is
   * expected and the {@link PlainTextDocumentReaderAndWriter} is used. Output
   * looks like: My/O name/O is/O Bill/PERSON Smith/PERSON ./O
   *
   * @param sentences The String to be classified
   * @return A String annotated with classification
   *         information.
   */
  public String classifyToString(String sentences) {
    return classifyToString(sentences, "slashTags", true);
  }

  /**
   * Classify the contents of a {@link String}.  Plain text or XML input text
   * is expected and the {@link PlainTextDocumentReaderAndWriter} is used.
   * Output is a (possibly empty, but not <code>null</code> List of Triples.
   * Each Triple is an entity name, followed by beginning and ending
   * character offsets in the original String.
   * Character offsets can be thought of as fenceposts between the characters,
   * or, like certain methods in the Java String class, as character positions,
   * numbered starting from 0, with the end index pointing to the position
   * AFTER the entity ends.  That is, end - start is the length of the entity
   * in characters.
   * <p>
   * <i>Fine points:</i> Token offsets are true wrt the source text, even though
   * the tokenizer may internally normalize certain tokens to String
   * representations of different lengths (e.g., " becoming `` or '').
   * When a period counts as both part of an abbreviation and as an end of
   * sentence marker, and that abbreviation is part of a named entity,
   * the reported entity string excludes the period.
   *
   * @param sentences The string to be classified
   * @return A {@link List} of {@link Triple}s, each of which gives an entity
   *     type and the beginning and ending character offsets.
   */
  public List<Triple<String,Integer,Integer>> classifyToCharacterOffsets(String sentences) {
    DocumentReaderAndWriter tmp = readerAndWriter;
    readerAndWriter = new PlainTextDocumentReaderAndWriter();
    readerAndWriter.init(flags);
    ObjectBank<List<CoreLabel>> documents = makeObjectBankFromString(sentences);
    readerAndWriter = tmp;

    List<Triple<String,Integer,Integer>> entities = new ArrayList<Triple<String,Integer,Integer>>();
    for (List<CoreLabel> doc : documents) {
      String prevEntityType = flags.backgroundSymbol;
      Triple<String,Integer,Integer> prevEntity = null;

      classify(doc);

      for (CoreLabel fl : doc) {
        String guessedAnswer = fl.get(AnswerAnnotation.class);
        if (guessedAnswer.equals(flags.backgroundSymbol)) {
          if (prevEntity != null) {
            entities.add(prevEntity);
            prevEntity = null;
          }
        } else {
          if ( ! guessedAnswer.equals(prevEntityType)) {
            if (prevEntity != null) {
              entities.add(prevEntity);
            }
            prevEntity = new Triple<String,Integer,Integer>(guessedAnswer, fl.get(BeginPositionAnnotation.class),
                fl.get(EndPositionAnnotation.class));
          } else {
            assert prevEntity != null; // if you read the code carefully, this should always be true!
            prevEntity.setThird(fl.get(EndPositionAnnotation.class));
          }
        }
        prevEntityType = guessedAnswer;
      }

      // include any entity at end of doc
      if (prevEntity != null) {
        entities.add(prevEntity);
      }

    }
    return entities;
  }


  /**
   * ONLY USE IF LOADED A CHINESE WORD SEGMENTER!!!!!
   *
   * @param sentence The string to be classified
   * @return List of words
   */
  public List<String> segmentString(String sentence) {
    ObjectBank<List<CoreLabel>> docs = makeObjectBankFromString(sentence);

    // @ cer  - previously, there was the following todo here:
    //
    //    TODO: use printAnswers(List<CoreLabel> doc, PrintWriter pw)
    //    instead
    //
    // I went ahead and did the TODO. However, given that the TODO
    // was incredibly easy to do, I'm wondering if it was left
    // as a todo for a reason. For example,  I'm concerned that something
    // else bizarrely breaks if this method calls printAnswers, as the method
    // arguably should, instead of manually building up the output string,
    // as was being done before.
    //
    // In any case, by doing the TODO, I was able to improve the online
    // parser/segmenter since all of the wonderful post processing
    // stuff is now being done to the segmented strings.
    //
    // However, if anything I'm not aware of broke, please just shot me
    // an e-mail (cerd@cs.colorado.edu) and I will look into and fix
    // the problem asap.

    // Also...
    //
    // Using a temporary file for flags.testFile is not elegant
    // However, I think all more elegant solutions would require
    // touching more source files. Touching more source files
    // risks incurring the wrath of whoever regularly works-with
    // and/or 'owns' this part of the codebase.
    //
    // (...the testFile stuff is necessary for segmentation whitespace
    //  normalization)

    String oldTestFile = flags.testFile;
    try {
      File tempFile = File.createTempFile("segmentString", ".txt");
      tempFile.deleteOnExit();
      flags.testFile = tempFile.getPath();
      FileWriter tempWriter = new FileWriter(tempFile);
      tempWriter.write(sentence);
      tempWriter.close();
    } catch (IOException e) {
      System.err.println("Warning(segmentString): " +
         "couldn't create temporary file for flags.testFile");
      flags.testFile = "";
    }

    StringWriter stringWriter = new StringWriter();
    PrintWriter stringPrintWriter = new PrintWriter(stringWriter);
    for (List<CoreLabel> doc : docs) {
      classify(doc);
      readerAndWriter.printAnswers(doc, stringPrintWriter);
      stringPrintWriter.println();
    }
    stringPrintWriter.close();
    String segmented = stringWriter.toString();

    flags.testFile = oldTestFile;
    return Arrays.asList(segmented.split("\\s"));
  }

  /**
   * Classify the contents of {@link SeqClassifierFlags scf.testFile}.
   * The file should be in the format
   * expected based on {@link SeqClassifierFlags scf.documentReader}.
   *
   * @return A {@link List} of {@link List}s of classified
   *         {@link CoreLabel}s where each
   *         {@link List} refers to a document/sentence.
   */
//   public ObjectBank<List<CoreLabel>> test() {
//     return test(flags.testFile);
//   }

  /**
   * Classify the contents of a file.  The file should be in the format
   * expected based on {@link SeqClassifierFlags scf.documentReader} if the
   * file is specified in {@link SeqClassifierFlags scf.testFile}.  If the
   * file being read is from {@link SeqClassifierFlags scf.textFile} then
   * the {@link PlainTextDocumentReaderAndWriter} is used.
   *
   * @param filename The path to the specified file
   * @return A {@link List} of {@link List}s of classified {@link CoreLabel}s where each
   *         {@link List} refers to a document/sentence.
   */
//   public ObjectBank<List<CoreLabel>> test(String filename) {
//     // only for the OCR data does this matter
//     flags.ocrTrain = false;

//     ObjectBank<List<CoreLabel>> docs = makeObjectBank(filename);
//     return testDocuments(docs);
//   }

  /**
   * Classify a {@link List} of {@link CoreLabel}s.
   *
   * @param document A {@link List} of {@link CoreLabel}s.
   * @return the same {@link List}, but with the elements annotated
   *         with their answers (with <code>setAnswer()</code>).
   */
  public abstract List<CoreLabel> classify(List<CoreLabel> document);



  /** Train the classifier based on values in flags.  It will use the first
   *  of these variables that is defined: trainFiles (and baseTrainDir),
   *  trainFileList, trainFile.
   */
  public void train() {
    if (flags.trainFiles != null) {
      train(flags.baseTrainDir, flags.trainFiles);
    } else if (flags.trainFileList != null) {
      String[] files = flags.trainFileList.split(",");
      train(files);
    } else {
      train(flags.trainFile);
    }
  }

  public void train(String filename) {
    // only for the OCR data does this matter
    flags.ocrTrain = true;
    train(makeObjectBankFromFile(filename));
  }

  public void train(String baseTrainDir, String trainFiles) {
    // only for the OCR data does this matter
    flags.ocrTrain = true;
    train(makeObjectBankFromFiles(baseTrainDir, trainFiles));
  }

  public void train(String[] trainFileList) {
    // only for the OCR data does this matter
    flags.ocrTrain = true;
    train(makeObjectBankFromFiles(trainFileList));
  }


  public abstract void train(ObjectBank<List<CoreLabel>> docs);


  /**
   * Reads a String into an ObjectBank object.
   * NOTE: that the current implementation of ReaderIteratorFactory will first
   * try to interpret each string as a filename, so this method
   * will yield unwanted results if it applies to a string that is
   * at the same time a filename. It prints out a warning, at least.
   *
   * @param string The String which will be the content of the ObjectBank
   *             (ASSUMING THAT NO FILE OF THIS NAME EXISTS!)
   * @return The ObjectBank
   */
  public ObjectBank<List<CoreLabel>> makeObjectBankFromString(String string) {
    // try to interpret as a file to throw warning.
    File file = new File(string);
    if (file.exists()) {
      System.err.println("Warning: calling makeObjectBankFromString with an existing file name! This will open the file instead.");
    }

    if (flags.announceObjectBankEntries) {
      System.err.print("Reading data using ");
      System.err.println(flags.readerAndWriter);

      if (flags.inputEncoding == null) {
        System.err.println("Getting data from " + string + " (default encoding)");
      } else {
        System.err.println("Getting data from " + string + " (" + flags.inputEncoding + " encoding)");
      }
    }

    return new ObjectBankWrapper(flags, new ObjectBank<List<CoreLabel>>(new ResettableReaderIteratorFactory(string), readerAndWriter), knownLCWords);
  }


  public ObjectBank<List<CoreLabel>> makeObjectBankFromFile(String filename) {
    String[] fileAsArray = {filename};
    return makeObjectBankFromFiles(fileAsArray);
  }


  public ObjectBank<List<CoreLabel>> makeObjectBankFromFiles(String[] trainFileList) {
    //try{
    Collection<File> files = new ArrayList<File>();
    for (String trainFile : trainFileList) {
      File f = new File(trainFile);
      files.add(f);
    }
    // System.err.printf("trainFileList contains %d file%s.\n", files.size(), files.size() == 1 ? "": "s");
    return new ObjectBankWrapper(flags, new ObjectBank<List<CoreLabel>>(new ResettableReaderIteratorFactory(files), readerAndWriter), knownLCWords);
    //} catch (IOException e) {
    //throw new RuntimeException(e);
    //}
  }


  public ObjectBank<List<CoreLabel>> makeObjectBankFromFiles(String baseDir, String filePattern) {
    try {
      File path = new File(baseDir);
      FileFilter filter = new RegExFileFilter(Pattern.compile(filePattern));
      File[] origFiles = path.listFiles(filter);
      Collection<BufferedReader> files = new ArrayList<BufferedReader>();
      for (File file : origFiles) {
        if (file.isFile()) {
          if (flags.inputEncoding == null) {
            if (flags.announceObjectBankEntries) {
              System.err.println("Getting data from " + file + " (default encoding)");
            }
            files.add(new BufferedReader(new InputStreamReader(new FileInputStream(file))));
          } else {
            if (flags.announceObjectBankEntries) {
              System.err.println("Getting data from " + file + " (" + flags.inputEncoding + " encoding)");
            }
            files.add(new BufferedReader(new InputStreamReader(new FileInputStream(file), flags.inputEncoding)));
          }
        }
      }

      if (files.isEmpty()) {
        throw new RuntimeException("No matching files: " + baseDir + '\t' + filePattern);
      }

      return new ObjectBankWrapper(flags, new ObjectBank<List<CoreLabel>>(new ResettableReaderIteratorFactory(files), readerAndWriter), knownLCWords);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }


  public ObjectBank<List<CoreLabel>> makeObjectBankFromFiles(Collection<File> files) {
    if (files.isEmpty()) {
        throw new RuntimeException("Attempt to make ObjectBank with empty file list");
    }

    return new ObjectBankWrapper(flags, new ObjectBank<List<CoreLabel>>(new ResettableReaderIteratorFactory(files), readerAndWriter), knownLCWords);
  }


  /** Set up an ObjectBank that will allow one to iterate over a
   *  collection of documents obtained from the passed in Reader.
   *  Each document will be represented as a list of CoreLabel.
   *  If the ObjectBank iterator() is called until hasNext() returns false,
   *  then the Reader will be read till end of file, but no
   *  reading is done at the time of this call.  Reading is done using the
   *  reading method specified in <code>flags.documentReader</code>,
   *  and for some reader choices, the column mapping given in
   *  <code>flags.map</code>.
   *
   * @param in      Input data
   * addNEWLCWords do we add new lowercase words from this data to the word shape classifier
   * @return The list of documents
   */
  protected ObjectBank<List<CoreLabel>> makeObjectBankFromReader(BufferedReader in) {
    if (flags.announceObjectBankEntries) {
      System.err.print("Reading data using ");
      System.err.println(flags.readerAndWriter);
    }

    return new ObjectBankWrapper(flags, new ObjectBank<List<CoreLabel>>(new ResettableReaderIteratorFactory(in), readerAndWriter), knownLCWords);
  }


  /**
   * Takes the file, reads it in, and prints out the likelihood of
   * each possible label at each point.
   *
   * @param filename The path to the specified file
   */
  public void printProbs(String filename) {
    // only for the OCR data does this matter
    flags.ocrTrain = false;

    ObjectBank<List<CoreLabel>> docs = makeObjectBankFromFile(filename);
    printProbsDocuments(docs);
  }

  /**
   * Takes a {@link List} of documents and prints the likelihood
   * of each possible label at each point.
   *
   * @param documents A {@link List} of {@link List} of {@link CoreLabel}s.
   */
  public void printProbsDocuments(ObjectBank<List<CoreLabel>> documents) {
    for (List<CoreLabel> doc : documents) {
      printProbsDocument(doc);
      System.out.println();
    }
  }

  public abstract void printProbsDocument(List<CoreLabel> document);


  /** Load a test file, run the classifier on it, and then print the answers
   *  to stdout (with timing to stderr).  This uses the value of
   *  flags.documentReader to determine testFile format.
   *
   *  @param testFile The file to test on.
   */
  public void classifyAndWriteAnswers(String testFile) throws Exception {
    ObjectBank<List<CoreLabel>> documents = makeObjectBankFromFile(testFile);
    classifyAndWriteAnswers(documents);
  }

  public void classifyAndWriteAnswers(String baseDir, String filePattern) throws Exception {
    ObjectBank<List<CoreLabel>> documents = makeObjectBankFromFiles(baseDir, filePattern);
    classifyAndWriteAnswers(documents);
  }

  public void classifyAndWriteAnswers(Collection<File> testFiles) throws Exception{
    ObjectBank<List<CoreLabel>> documents = makeObjectBankFromFiles(testFiles);
    classifyAndWriteAnswers(documents);
  }

  private void classifyAndWriteAnswers(ObjectBank<List<CoreLabel>> documents) throws Exception {
    Timing timer = new Timing();
    int numWords = 0;
    int numDocs = 0;
    for (List<CoreLabel> doc : documents) {
      classify(doc);
      numWords += doc.size();
      writeAnswers(doc);
      numDocs++;
    }
    long millis = timer.stop();
    double wordspersec = numWords / (((double) millis) / 1000);
    NumberFormat nf = new DecimalFormat("0.00"); // easier way!
    System.err.println(StringUtils.getShortClassName(this) +
                       " tagged " + numWords + " words in " + numDocs +
                       " documents at " + nf.format(wordspersec) +
                       " words per second.");
  }


  /** Load a test file, run the classifier on it, and then print the answers
   *  to stdout (with timing to stderr).  This uses the value of
   *  flags.documentReader to determine testFile format.
   *
   *  @param testFile The file to test on.
   */
  public void classifyAndWriteAnswersKBest(String testFile, int k) throws Exception {
    Timing timer = new Timing();
    ObjectBank<List<CoreLabel>> documents = makeObjectBankFromFile(testFile);
    int numWords = 0;
    int numSentences = 0;

    for (List<CoreLabel> doc : documents) {
      Counter<List<CoreLabel>> kBest = classifyKBest(doc, AnswerAnnotation.class, k);
      numWords += doc.size();
      List<List<CoreLabel>> sorted = Counters.toSortedList(kBest);
      int n = 1;
      for (List<CoreLabel> l : sorted) {
        System.out.println("<sentence id="+numSentences+" k="+n+" logProb="+kBest.getCount(l)+" prob="+Math.exp(kBest.getCount(l))+ '>');
        writeAnswers(l);
        System.out.println("</sentence>");
        n++;
      }
      numSentences++;
    }

    long millis = timer.stop();
    double wordspersec = numWords / (((double) millis) / 1000);
    NumberFormat nf = new DecimalFormat("0.00"); // easier way!
    System.err.println(this.getClass().getName()+" tagged " + numWords + " words in " + numSentences +
                       " documents at " + nf.format(wordspersec) +
                       " words per second.");
  }

  /** Load a test file, run the classifier on it, and then write a Viterbi search graph for
   *  each sequence.
   *
   *  @param testFile The file to test on.
   */
  public void classifyAndWriteViterbiSearchGraph(String testFile, String searchGraphPrefix)
       throws Exception {
    Timing timer = new Timing();
    ObjectBank<List<CoreLabel>> documents = makeObjectBankFromFile(testFile);
    int numWords = 0;
    int numSentences = 0;

    for (List<CoreLabel> doc : documents) {
      DFSA<String, Integer> tagLattice = getViterbiSearchGraph(doc, AnswerAnnotation.class);
      numWords += doc.size();
      PrintWriter latticeWriter = new PrintWriter(new FileOutputStream(searchGraphPrefix+ '.' +numSentences+".wlattice"));
      PrintWriter vsgWriter = new PrintWriter(new FileOutputStream(searchGraphPrefix+ '.' +numSentences+".lattice"));
      if(readerAndWriter instanceof LatticeWriter)
        ((LatticeWriter)readerAndWriter).printLattice(tagLattice, doc, latticeWriter);
      tagLattice.printAttFsmFormat(vsgWriter);
      latticeWriter.close();
      vsgWriter.close();
      numSentences++;
    }

    long millis = timer.stop();
    double wordspersec = numWords / (((double) millis) / 1000);
    NumberFormat nf = new DecimalFormat("0.00"); // easier way!
    System.err.println(this.getClass().getName()+" tagged " + numWords + " words in " + numSentences +
                       " documents at " + nf.format(wordspersec) +
                       " words per second.");
  }

  /** Write the classifications of the Sequence classifier out
   *  to stdout in a format
   *  determined by the DocumentReaderAndWriter used.
   *  If the flag <code>outputEncoding</code> is defined, the output
   *  is written in that character encoding, otherwise in the system default
   *  character encoding.
   *
   *  @param doc Documents to write out
   *  @throws Exception If an IO problem
   */
  public void writeAnswers(List<CoreLabel> doc) throws Exception {
    if (flags.lowerNewgeneThreshold) {
      return;
    }
    if (flags.numRuns <= 1) {
      PrintWriter out;
      if (flags.outputEncoding == null) {
        out = new PrintWriter(System.out, true);
      } else {
        out = new PrintWriter(new OutputStreamWriter(System.out, flags.outputEncoding), true);
      }
      readerAndWriter.printAnswers(doc, out);
//      out.println();
      out.flush();
    }
  }


  /** Serialize a sequence classifier to a file on the given path.
   *
   *  @param serializePath The path/filename to write the classifier to.
   */
  public abstract void serializeClassifier(String serializePath);


  /**
   * Loads a classifier from the given input stream.
   * The JVM shuts down (System.exit(1)) if there is an exception.
   * This does not close the InputStream.
   *
   * @param in The InputStream to read from
   */
  public void loadClassifierNoExceptions(InputStream in) {
    // load the classifier
    try {
      loadClassifier(in);
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }

  }

  /** Load a classsifier from the specified InputStream.
   *  No extra properties are supplied.
   *  This does not close the InputStream.
   *
   *  @param in The InputStream to load the serialized classifier from
   *
   *  @throws IOException If there are problems accessing the input stream
   *  @throws ClassCastException If there are problems interpreting the serialized data
   *  @throws ClassNotFoundException If there are problems interpreting the serialized data
   */
  public void loadClassifier(InputStream in) throws IOException, ClassCastException, ClassNotFoundException {
    loadClassifier(in, null);
  }

  /** Load a classsifier from the specified InputStream.
   *  The classifier is reinitialized from the flags serialized in the
   *  classifier.
   *  This does not close the InputStream.
   *
   *  @param in The InputStream to load the serialized classifier from
   *  @param props This Properties object will be used to update the SeqClassifierFlags which
   *               are read from the serialized classifier
   *
   *  @throws IOException If there are problems accessing the input stream
   *  @throws ClassCastException If there are problems interpreting the serialized data
   *  @throws ClassNotFoundException If there are problems interpreting the serialized data
   */
  public void loadClassifier(InputStream in, Properties props) throws IOException, ClassCastException, ClassNotFoundException {
    loadClassifier(new ObjectInputStream(in), props);
  }

  /** Load a classsifier from the specified input stream.
   *  The classifier is reinitialized from the flags serialized in the
   *  classifier.
   *
   *  @param in The InputStream to load the serialized classifier from
   *  @param props This Properties object will be used to update the SeqClassifierFlags which
   *               are read from the serialized classifier
   *
   *  @throws IOException If there are problems accessing the input stream
   *  @throws ClassCastException If there are problems interpreting the serialized data
   *  @throws ClassNotFoundException If there are problems interpreting the serialized data
   */
  public abstract void loadClassifier(ObjectInputStream in, Properties props) throws IOException, ClassCastException, ClassNotFoundException;

  /**
   * Loads a classifier from the file specified by loadPath.  If loadPath
   * ends in .gz, uses a GZIPInputStream, else uses a regular FileInputStream.
   */
  public void loadClassifier(String loadPath) throws ClassCastException, IOException, ClassNotFoundException {
    loadClassifier(new File(loadPath));
  }

  public void loadClassifierNoExceptions(String loadPath) {
    loadClassifierNoExceptions(new File(loadPath));
  }

  public void loadClassifierNoExceptions(String loadPath, Properties props) {
    loadClassifierNoExceptions(new File(loadPath), props);
  }

  public void loadClassifier(File file) throws ClassCastException, IOException, ClassNotFoundException {
    loadClassifier(file, null);
  }

  /**
   * Loads a classifier from the file specified.  If the file's name
   * ends in .gz, uses a GZIPInputStream, else uses a regular FileInputStream.
   * This method closes the File when done.
   *
   * @param file Loads a classifier from this file.
   * @param props Properties in this object will be used to overwrite those
   *         specified in the serialized classifier
   *
   * @throws IOException If there are problems accessing the input stream
   * @throws ClassCastException If there are problems interpreting the serialized data
   * @throws ClassNotFoundException If there are problems interpreting the serialized data
   */
  public void loadClassifier(File file, Properties props) throws ClassCastException, IOException, ClassNotFoundException {
    Timing.startDoing("Loading classifier from " + file.getAbsolutePath());
    BufferedInputStream bis;
    if (file.getName().endsWith(".gz")) {
      bis = new BufferedInputStream(new GZIPInputStream(new FileInputStream(file)));
    } else {
      bis = new BufferedInputStream(new FileInputStream(file));
    }
    loadClassifier(bis, props);
    bis.close();
    Timing.endDoing();
  }


  public void loadClassifierNoExceptions(File file) {
    loadClassifierNoExceptions(file, null);
  }

  public void loadClassifierNoExceptions(File file, Properties props) {
    try {
      loadClassifier(file, props);
    } catch (Exception e) {
      System.err.println("Error deserializing " + file.getAbsolutePath());
      e.printStackTrace();
      System.exit(1);
    }
  }

  /**
   * This function will load a classifier that is stored inside a jar file
   * (if it is so stored).  The classifier should be specified as its full
   * filename, but the path in the jar file (<code>/classifiers/</code>) is
   * coded in this class.  If the classifier is not stored in the jar file
   * or this is not run from inside a jar file, then this function will
   * throw a RuntimeException.
   *
   * @param modelName The name of the model file.  Iff it ends in .gz, then
   *             it is assumed to be gzip compressed.
   * @param props A Properties object which can override certain properties
   *             in the serialized file, such as the DocumentReaderAndWriter.
   *             You can pass in <code>null</code> to override nothing.
   */
  public void loadJarClassifier(String modelName, Properties props) {
    Timing.startDoing("Loading JAR-internal classifier " + modelName);
    try {
      InputStream is = getClass().getResourceAsStream(JAR_CLASSIFIER_PATH + modelName);
      if (modelName.endsWith(".gz")) {
        is = new GZIPInputStream(is);
      }
      is = new BufferedInputStream(is);
      loadClassifier(is, props);
      is.close();
      Timing.endDoing();
    } catch (Exception e) {
      String msg = "Error loading classifier from jar file (most likely you are not running this code from a jar file or the named classifier is not stored in the jar file)";
      throw new RuntimeException(msg, e);
    }
  }

}
