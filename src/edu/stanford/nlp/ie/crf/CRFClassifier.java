// CRFClassifier -- a probabilistic (CRF) sequence model, mainly used for NER.
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

package edu.stanford.nlp.ie.crf;

import edu.stanford.nlp.ie.*;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations.AnswerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.maxent.Convert;
import edu.stanford.nlp.objectbank.ObjectBank;
import edu.stanford.nlp.optimization.*;
import edu.stanford.nlp.sequences.*;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.PaddedList;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Class for Sequence Classification using a Conditional Random Field model.
 * The code has functionality for different document formats, but when
 * using the standard {@link ColumnDocumentReaderAndWriter} for training
 * or testing models, input files are expected to
 * be one token per line with the columns indicating things like the word,
 * POS, chunk, and answer class.  The default for
 * <code>ColumnDocumentReaderAndWriter</code> training data is 3 column input,
 * with the columns containing a word, its POS, and its gold class, but
 * this can be specified via the <code>map</code> property.
 * <p/>
 * When run on a file with <code>-textFile</code>,
 * the file is assumed to be plain English text (or perhaps simple HTML/XML),
 * and a reasonable attempt is made at English tokenization by
 * {@link PlainTextDocumentReaderAndWriter}.
 * <p/>
 * <b>Typical command-line usage</b>
 * <p>For running a trained model with a provided serialized classifier on a
 * text file: <p>
 * <code>
 * java -mx500m edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier
 * conll.ner.gz -textFile samplesentences.txt
 * </code><p>
 * When specifying all parameters in a properties file (train, test, or
 * runtime):<p>
 * <code>
 * java -mx1g edu.stanford.nlp.ie.crf.CRFClassifier -prop propFile
 * </code><p>
 * To train and test a simple NER model from the command line:<p>
 * <code>java -mx1000m edu.stanford.nlp.ie.crf.CRFClassifier
 * -trainFile trainFile -testFile testFile -macro &gt; output </code>
 * <p/>
 * Features are defined by a {@link edu.stanford.nlp.sequences.FeatureFactory}.
 * {@link NERFeatureFactory} is used by default, and
 * you should look there for feature templates and properties or flags that
 * will cause certain features to be used when training an NER classifier.
 * There is also
 * a {@link edu.stanford.nlp.wordseg.SighanFeatureFactory}, and various
 * successors such as
 * {@link edu.stanford.nlp.wordseg.ChineseSegmenterFeatureFactory},
 * which are used for Chinese word segmentation.
 * Features are specified either by a Properties file (which is the
 * recommended method) or by flags on the command line.  The flags are read
 * into a {@link SeqClassifierFlags} object, which the
 * user need not be concerned with, unless wishing to add new features.
 * <p/>
 * CRFClassifier may also be used programatically.  When creating a new
 * instance, you <i>must</i> specify a Properties object.  You may then
 * call train methods to train a classifier, or load a classifier.
 * The other way to get a CRFClassifier is to deserialize one via
 * the static {@link CRFClassifier#getClassifier(String)} methods, which
 * return a deserialized
 * classifier.  You may then tag (classify the items of) documents
 * using either the assorted
 * <code>classify()</code> or the assorted <code>classify</code> methods in
 * {@link AbstractSequenceClassifier}.
 * Probabilities assigned by the CRF can be interrogated using either the
 * <code>printProbsDocument()</code> or
 * <code>getCliqueTrees()</code> methods.
 *
 * @author Jenny Finkel
 */
public class CRFClassifier extends AbstractSequenceClassifier {

  Index<CRFLabel>[] labelIndices;
  /** Parameter weights of the classifier. */
  double[][] weights;
  Index<String> featureIndex;
  int[] map;  // caches the featureIndex
  Minimizer minimizer;

  /** Name of default serialized classifier resource to look for in a jar file.
   */
  public static final String DEFAULT_CLASSIFIER = "ner-eng-ie.crf-3-all2008.ser.gz";
  private static final boolean VERBOSE = false;

  // List selftraindatums = new ArrayList();


  protected CRFClassifier() {
    super(new SeqClassifierFlags());
  }

  public CRFClassifier(Properties props) {
    super(props);
  }

  public void dropFeaturesBelowThreshold(double threshold) {
    Index<String> newFeatureIndex = new Index<String>();
    for (int i = 0; i < weights.length; i++) {
      double smallest = weights[i][0];
      double biggest = weights[i][0];
      for (int j = 1; j < weights[i].length; j++) {
        if (weights[i][j] > biggest) {
          biggest = weights[i][j];
        }
        if (weights[i][j] < smallest) {
          smallest = weights[i][j];
        }
        if (biggest - smallest > threshold) {
          newFeatureIndex.add(featureIndex.get(i));
          break;
        }
      }
    }

    int[] newMap = new int[newFeatureIndex.size()];
    for (int i = 0; i < newMap.length; i++) {
      int index = featureIndex.indexOf(newFeatureIndex.get(i));
      newMap[i] = map[index];
    }
    map = newMap;
    featureIndex = newFeatureIndex;
  }

  /**
   * Convert a document List into arrays storing the data features and labels.
   *
   * @param document Training documents
   * @return A Pair, where the first element is an int[][][] representing the data
   *         and the second element is an int[] representing the labels
   */
  public Pair<int[][][],int[]> documentToDataAndLabels(List<? extends CoreLabel> document) {

    int docSize = document.size();
    // first index is position in the document also the index of the clique/factor table
    // second index is the number of elements in the clique/window these features are for (starting with last element)
    // third index is position of the feature in the array that holds them
    // element in data[j][k][m] is the index of the mth feature occurring in position k of the jth clique
    int[][][] data = new int[docSize][windowSize][];
    // index is the position in the document
    // element in labels[j] is the index of the correct label (if it exists) at position j of document
    int[] labels = new int[docSize];

    if (flags.useReverse) {
      Collections.reverse(document);
    }

    //System.err.println("docSize:"+docSize);
    for (int j = 0; j < docSize; j++) {
      CRFDatum d = makeDatum(document, j, featureFactory);

      List features = d.asFeatures();
      for (int k = 0, fSize = features.size(); k < fSize; k++) {
        Collection<String> cliqueFeatures = (Collection<String>) features.get(k);
        data[j][k] = new int[cliqueFeatures.size()];
        int m = 0;
        for (String feature : cliqueFeatures) {
          int index = featureIndex.indexOf(feature);
          if (index >= 0) {
            data[j][k][m] = index;
            m++;
          } else {
            // this is where we end up when we do feature threshhold cutoffs
          }
        }
        // Reduce memory use when some feaures were cut out by threshold
        if (m < data[j][k].length) {
          int[] f = new int[m];
          System.arraycopy(data[j][k], 0, f, 0, m);
          data[j][k] = f;
        }
      }

      CoreLabel wi = document.get(j);
      labels[j] = classIndex.indexOf(wi.get(AnswerAnnotation.class));
    }

    if (flags.useReverse) {
      Collections.reverse(document);
    }

    // 	System.err.println("numClasses: "+classIndex.size()+" "+classIndex);
    // 	System.err.println("numDocuments: 1");
    // 	System.err.println("numDatums: "+data.length);
    // 	System.err.println("numFeatures: "+featureIndex.size());

    return new Pair<int[][][],int[]>(data, labels);
  }


  public void printLabelInformation(String testFile) throws Exception {
    ObjectBank<List<CoreLabel>> documents = makeObjectBankFromFile(testFile);
    for (List<CoreLabel> document : documents) {
      printLabelValue(document);
    }
  }


  public void printLabelValue(List<CoreLabel> document) {

    if (flags.useReverse) {
      Collections.reverse(document);
    }

    NumberFormat nf = new DecimalFormat();

    List<String> classes = new ArrayList<String>();
    for (int i = 0; i < classIndex.size(); i++) {
      classes.add(classIndex.get(i));
    }
    String[] columnHeaders = classes.toArray(new String[classes.size()]);

    //System.err.println("docSize:"+docSize);
    for (int j = 0; j < document.size(); j++) {

      System.out.println("--== "+document.get(j).get(WordAnnotation.class)+" ==--");

      List<String[]> lines = new ArrayList<String[]>();
      List<String> rowHeaders = new ArrayList<String>();
      List<String> line = new ArrayList<String>();

      for (int p = 0; p < labelIndices.length; p++) {
        if (j+p >= document.size()) { continue; }
        CRFDatum d = makeDatum(document, j+p, featureFactory);

        List features = d.asFeatures();
        for (int k = p, fSize = features.size(); k < fSize; k++) {
          Collection<String> cliqueFeatures = (Collection<String>) features.get(k);
          for (String feature : cliqueFeatures) {
            int index = featureIndex.indexOf(feature);
            if (index >= 0) {
//              line.add(feature+"["+(-p)+"]");
              rowHeaders.add(feature+ '[' + (-p) + ']');
              double[] values = new double[labelIndices[0].size()];
              for (CRFLabel label : labelIndices[k]) {
                int[] l = label.getLabel();
                double v = weights[index][labelIndices[k].indexOf(label)];
                values[l[l.length-1-p]] += v;
              }
              for (double value : values) {
                line.add(nf.format(value));
              }
              lines.add(line.toArray(new String[line.size()]));
              line = new ArrayList<String>();
            }
          }
        }
//        lines.add(Collections.<String>emptyList());
        System.out.println(StringUtils.makeAsciiTable(lines.toArray(new String[lines.size()][0]),
                rowHeaders.toArray(new String[rowHeaders.size()]),
                columnHeaders,
                0, 1, true));
        System.out.println();
      }
//      System.err.println(edu.stanford.nlp.util.StringUtils.join(lines,"\n"));
    }

    if (flags.useReverse) {
      Collections.reverse(document);
    }
  }


  /** Convert an ObjectBank to arrays of data features and labels.
   *
   * @param documents
   * @return A Pair, where the first element is an int[][][][] representing the data
   *    and the second element is an int[][] representing the labels.
   */
  public Pair<int[][][][],int[][]> documentsToDataAndLabels(ObjectBank<List<CoreLabel>> documents) {

    // first index is the number of the document
    // second index is position in the document also the index of the clique/factor table
    // third index is the number of elements in the clique/window thase features are for (starting with last element)
    // fourth index is position of the feature in the array that holds them
    // element in data[i][j][k][m] is the index of the mth feature occurring in position k of the jth clique of the ith document
    //    int[][][][] data = new int[documentsSize][][][];
    List<int[][][]> data = new ArrayList<int[][][]>();

    // first index is the number of the document
    // second index is the position in the document
    // element in labels[i][j] is the index of the correct label (if it exists) at position j in document i
    //    int[][] labels = new int[documentsSize][];
    List<int[]> labels = new ArrayList<int[]>();

    int numDatums = 0;

    for (List<CoreLabel> doc : documents) {
      Pair<int[][][],int[]> docPair = documentToDataAndLabels(doc);
      data.add(docPair.first());
      labels.add(docPair.second());
      numDatums += doc.size();
    }

    System.err.println("numClasses: " + classIndex.size() + " " + classIndex);
    System.err.println("numDocuments: " + data.size());
    System.err.println("numDatums: " + numDatums);
    System.err.println("numFeatures: " + featureIndex.size());
    printFeatures();

    int[][][][] dataA = new int[0][][][];
    int[][] labelsA = new int[0][];

    return new Pair<int[][][][],int[][]>(data.toArray(dataA), labels.toArray(labelsA));
  }

  private void printFeatures() {
    if (flags.printFeatures == null) {
      return;
    }
    try {
      String enc = flags.inputEncoding;
      if (flags.inputEncoding == null) {
        System.err.println("flags.inputEncoding doesn't exist, Use UTF-8 as default");
        enc = "UTF-8";
      }

      PrintWriter pw = new PrintWriter(new OutputStreamWriter(
          new FileOutputStream("feats-" + flags.printFeatures + ".txt"), enc), true);
      for (int i = 0; i < featureIndex.size(); i++) {
        pw.println(featureIndex.get(i));
      }
      pw.close();
    } catch (IOException ioe) {
      ioe.printStackTrace();
    }
  }

  /** This routine builds the <code>labelIndices</code> which give the
   *  empirically legal label sequences (of length (order) at most
   *  <code>windowSize</code>)
   *  and the <code>classIndex</code>,
   *  which indexes known answer classes.
   *
   * @param ob The training data: Read from an ObjectBank, each
   *                  item in it is a List<CoreLabel>.
   */
  private void makeAnswerArraysAndTagIndex(ObjectBank<List<CoreLabel>> ob) {

    HashSet<String>[] featureIndices = new HashSet[windowSize];
    for (int i = 0; i < windowSize; i++) {
      featureIndices[i] = new HashSet<String>();
    }

    labelIndices = new Index[windowSize];
    for (int i = 0; i < labelIndices.length; i++) {
      labelIndices[i] = new Index<CRFLabel>();
    }

    Index<CRFLabel> labelIndex = labelIndices[windowSize - 1];

    classIndex = new Index<String>();
    //classIndex.add("O");
    classIndex.add(flags.backgroundSymbol);

    HashSet[] seenBackgroundFeatures = new HashSet[2];
    seenBackgroundFeatures[0] = new HashSet();
    seenBackgroundFeatures[1] = new HashSet();

    //int count = 0;
    for (List<CoreLabel> doc : ob) {
      //if (count % 100 == 0) {
      //System.err.println(count);
      //}
      //count++;

      if (flags.useReverse) {
        Collections.reverse(doc);
      }

      int docSize = doc.size();
      //create the full set of labels in classIndex
      //note: update to use addAll later
      for (int j = 0; j < docSize; j++) {
        String ans = doc.get(j).get(AnswerAnnotation.class);
        classIndex.add(ans);
      }

      for (int j = 0; j < docSize; j++) {

        CRFDatum<Serializable,CRFLabel> d = makeDatum(doc, j, featureFactory);
        labelIndex.add(d.label());

        List<Serializable> features = d.asFeatures();
        for (int k = 0, fsize = features.size(); k < fsize; k++) {
          Collection<String> cliqueFeatures = (Collection<String>) features.get(k);
          if (k < 2 && flags.removeBackgroundSingletonFeatures) {
            String ans = doc.get(j).get(AnswerAnnotation.class);
            boolean background = ans.equals(flags.backgroundSymbol);
            if (k == 1 && j > 0 && background) {
              ans = doc.get(j - 1).get(AnswerAnnotation.class);
              background = ans.equals(flags.backgroundSymbol);
            }
            if (background) {
              for (String f : cliqueFeatures) {
                if (!featureIndices[k].contains(f)) {
                  if (seenBackgroundFeatures[k].contains(f)) {
                    seenBackgroundFeatures[k].remove(f);
                    featureIndices[k].add(f);
                  } else {
                    seenBackgroundFeatures[k].add(f);
                  }
                }
              }
            } else {
              seenBackgroundFeatures[k].removeAll(cliqueFeatures);
              featureIndices[k].addAll(cliqueFeatures);
            }
          } else {
            featureIndices[k].addAll(cliqueFeatures);
          }
        }
      }

      if (flags.useReverse) {
        Collections.reverse(doc);
      }
    }

    // String[] fs = new String[featureIndices[0].size()];
    // for (Iterator iter = featureIndices[0].iterator(); iter.hasNext(); ) {
    //   System.err.println(iter.next());
    // }

    int numFeatures = 0;
    for (int i = 0; i < windowSize; i++) {
      numFeatures += featureIndices[i].size();
    }

    featureIndex = new Index<String>();
    map = new int[numFeatures];
    for (int i = 0; i < windowSize; i++) {
      featureIndex.addAll(featureIndices[i]);
      for (String str : featureIndices[i]) {
        map[featureIndex.indexOf(str)] = i;
      }
    }

    if (flags.useObservedSequencesOnly) {
      for (int i = 0, liSize = labelIndex.size(); i < liSize; i++) {
        CRFLabel label = labelIndex.get(i);
        for (int j = windowSize - 2; j >= 0; j--) {
          label = label.getOneSmallerLabel();
          labelIndices[j].add(label);
        }
      }
    } else {
      for (int i = 0; i < labelIndices.length; i++) {
        labelIndices[i] = allLabels(i + 1, classIndex);
      }
    }

    if (VERBOSE) {
      for (int i = 0, fiSize = featureIndex.size(); i < fiSize; i++) {
        System.out.println(i + ": " + featureIndex.get(i));
      }
    }
  }

  protected static Index<CRFLabel> allLabels(int window, Index classIndex) {
    int[] label = new int[window];
    // cdm july 2005: below array initialization isn't necessary: JLS (3rd ed.) 4.12.5
    // Arrays.fill(label, 0);
    int numClasses = classIndex.size();
    Index<CRFLabel> labelIndex = new Index<CRFLabel>();
    OUTER: while (true) {
      CRFLabel l = new CRFLabel(label);
      labelIndex.add(l);
      int[] label1 = new int[window];
      System.arraycopy(label, 0, label1, 0, label.length);
      label = label1;
      for (int j = 0; j < label.length; j++) {
        label[j]++;
        if (label[j] >= numClasses) {
          label[j] = 0;
          if (j == label.length - 1) {
            break OUTER;
          }
        } else {
          break;
        }
      }
    }
    return labelIndex;
  }


  /** Makes a CRFDatum by producing features and a label from input data
   *  at a specific position, using the provided factory.
   *  @param info The input data
   *  @param loc The position to build a datum at
   *  @param featureFactory The FeatureFactory to use to extract features
   *  @return The constructed CRFDatum
   */
  public CRFDatum<Serializable,CRFLabel> makeDatum(List<? extends CoreLabel> info, int loc, edu.stanford.nlp.sequences.FeatureFactory featureFactory) {
    pad.set(AnswerAnnotation.class, flags.backgroundSymbol);
    PaddedList<? extends CoreLabel> pInfo = new PaddedList<CoreLabel>((List<CoreLabel>)info, pad);

    ArrayList features = new ArrayList();

    //     for (int i = 0; i < windowSize; i++) {
    //       List featuresC = new ArrayList();
    //       for (int j = 0; j < FeatureFactory.win[i].length; j++) {
    //         featuresC.addAll(featureFactory.features(info, loc, FeatureFactory.win[i][j]));
    //       }
    //       features.add(featuresC);
    //     }

    Collection<Clique> done = new HashSet<Clique>();
    for (int i = 0; i < windowSize; i++) {
      List featuresC = new ArrayList();
      Collection<Clique> windowCliques = featureFactory.getCliques(i, 0);
      windowCliques.removeAll(done);
      done.addAll(windowCliques);
      for (Clique c : windowCliques) {
        featuresC.addAll(featureFactory.getCliqueFeatures(pInfo, loc, c));
      }
      features.add(featuresC);
    }

    int[] labels = new int[windowSize];

    for (int i = 0; i < windowSize; i++) {
      String answer = pInfo.get(loc + i - windowSize + 1).get(AnswerAnnotation.class);

      labels[i] = classIndex.indexOf(answer);
    }

    CRFDatum<Serializable,CRFLabel> d = new CRFDatum<Serializable,CRFLabel>(features, new CRFLabel(labels));
    //System.err.println(d);
    return d;
  }


  public static class TestSequenceModel implements SequenceModel {

    private int window;
    private int numClasses;
    //private FactorTable[] factorTables;
    private CRFCliqueTree cliqueTree;
    private int[] tags;
    private int[] backgroundTag;

    //public Scorer(FactorTable[] factorTables) {
    public TestSequenceModel(CRFCliqueTree cliqueTree) {
      //this.factorTables = factorTables;
      this.cliqueTree = cliqueTree;
      //this.window = factorTables[0].windowSize();
      this.window = cliqueTree.window();
      //this.numClasses = factorTables[0].numClasses();
      this.numClasses = cliqueTree.getNumClasses();
      tags = new int[numClasses];
      for (int i = 0; i < tags.length; i++) {
        tags[i] = i;
      }
      backgroundTag = new int[]{cliqueTree.backgroundIndex()};
    }

    public int length() {
      return cliqueTree.length();
    }

    public int leftWindow() {
      return window - 1;
    }

    public int rightWindow() {
      return 0;
    }

    public int[] getPossibleValues(int pos) {
      if (pos < window - 1) {
        return backgroundTag;
      }
      return tags;
    }

    public double scoreOf(int[] tags, int pos) {
      int[] previous = new int[window - 1];
      int realPos = pos - window + 1;
      for (int i = 0; i < window - 1; i++) {
        previous[i] = tags[realPos + i];
      }
      return cliqueTree.condLogProbGivenPrevious(realPos, tags[pos], previous);
    }

    public double[] scoresOf(int[] tags, int pos) {
      int realPos = pos - window + 1;
      double[] scores = new double[numClasses];
      int[] previous = new int[window - 1];
      for (int i = 0; i < window - 1; i++) {
        previous[i] = tags[realPos + i];
      }
      for (int i = 0; i < numClasses; i++) {
        scores[i] = cliqueTree.condLogProbGivenPrevious(realPos, i, previous);
      }
      return scores;
    }

    public double scoreOf(int[] sequence) {
      throw new UnsupportedOperationException();
    }

  } // end class TestSequenceModel


  @Override
  public List<CoreLabel> classify(List<CoreLabel> document) {
    if (flags.doGibbs) {
      try {
        return classifyGibbs(document);
      } catch (Exception e) {
        System.err.println("Error running testGibbs inference!");
        e.printStackTrace();
        return null;
      }
    } else if (flags.crfType.equalsIgnoreCase("maxent")) {
      return classifyMaxEnt(document);
    } else {
      throw new RuntimeException("Unsupported inference type: " + flags.crfType);
    }
  }

  @Override
  public SequenceModel getSequenceModel(List<? extends CoreLabel> doc) {
    Pair<int[][][],int[]> p = documentToDataAndLabels(doc);
    int[][][] data = p.first();

    CRFCliqueTree cliqueTree = CRFCliqueTree.getCalibratedCliqueTree(weights, data, labelIndices, classIndex.size(), classIndex, flags.backgroundSymbol);

    //Scorer scorer = new Scorer(factorTables);
    return new TestSequenceModel(cliqueTree);
  }

  /** Do standard sequence inference, using either Viterbi or Beam inference
   *  depending on the value of <code>flags.inferenceType</code>.
   *
   *  @param document Document to classify. Classification happens in place.
   *             This document is modified.
   *  @return The classified document
   */
  public List<CoreLabel> classifyMaxEnt(List<CoreLabel> document) {
    if (document.isEmpty()) {
      return document;
    }

    SequenceModel model = getSequenceModel(document);

    if (flags.inferenceType == null) { flags.inferenceType = "Viterbi"; }

    BestSequenceFinder tagInference;
    if (flags.inferenceType.equalsIgnoreCase("Viterbi")) {
      tagInference = new ExactBestSequenceFinder();
    } else if (flags.inferenceType.equalsIgnoreCase("Beam")) {
      tagInference = new BeamBestSequenceFinder(flags.beamSize);
    } else {
      throw new RuntimeException("Unknown inference type: "+flags.inferenceType+". Your options are Viterbi|Beam.");
    }

    int[] bestSequence = tagInference.bestSequence(model);

    if (flags.useReverse) {
      Collections.reverse(document);
    }
    for (int j = 0, docSize = document.size(); j < docSize; j++) {
      CoreLabel wi = document.get(j);
      String guess = classIndex.get(bestSequence[j + windowSize - 1]);
      wi.set(AnswerAnnotation.class, guess);
    }
    if (flags.useReverse) {
      Collections.reverse(document);
    }
    return document;
  }


  public List<CoreLabel> classifyGibbs(List<CoreLabel> document) throws ClassNotFoundException, SecurityException, NoSuchMethodException, IllegalArgumentException, InstantiationException, IllegalAccessException, InvocationTargetException  {
    System.err.println("Testing using Gibbs sampling.");
    Pair<int[][][],int[]> p = documentToDataAndLabels(document);
    int[][][] data = p.first();

    List<CoreLabel> newDocument = document; // reversed if necessary
    if (flags.useReverse) {
      Collections.reverse(document);
      newDocument = new ArrayList<CoreLabel>(document);
      Collections.reverse(document);
    }

    CRFCliqueTree cliqueTree = CRFCliqueTree.getCalibratedCliqueTree(weights, data, labelIndices, classIndex.size(), classIndex, flags.backgroundSymbol);

    SequenceModel model = cliqueTree;
    SequenceListener listener = cliqueTree;

    EntityCachingAbstractSequencePrior prior;

    if (flags.useNERPrior) {
      prior = new EmpiricalNERPrior(flags.backgroundSymbol, classIndex, newDocument);
      // SamplingNERPrior prior = new SamplingNERPrior(flags.backgroundSymbol, classIndex, newDocument);
    } else if (flags.useAcqPrior) {
      prior = new AcquisitionsPrior(flags.backgroundSymbol, classIndex, newDocument);
    } else if (flags.useSemPrior) {
      prior = new SeminarsPrior(flags.backgroundSymbol, classIndex, newDocument);
    } else {
      throw new RuntimeException("no prior specified");
    }

    model = new FactoredSequenceModel(model, prior);
    listener = new FactoredSequenceListener(listener, prior);

    SequenceGibbsSampler sampler = new SequenceGibbsSampler(0, 0, listener);
    int[] sequence = new int[cliqueTree.length()];

    if (flags.initViterbi) {
      TestSequenceModel testSequenceModel = new TestSequenceModel(cliqueTree);
      ExactBestSequenceFinder tagInference = new ExactBestSequenceFinder();
      int[] bestSequence = tagInference.bestSequence(testSequenceModel);
      System.arraycopy(bestSequence, windowSize-1, sequence, 0, sequence.length);
    } else {
      int[] initialSequence = SequenceGibbsSampler.getRandomSequence(model);
      System.arraycopy(initialSequence, 0, sequence, 0, sequence.length);
    }

    sampler.verbose = 0;

    if (flags.annealingType.equalsIgnoreCase("linear")) {
      sequence = sampler.findBestUsingAnnealing(model, CoolingSchedule.getLinearSchedule(1.0, flags.numSamples), sequence);
    } else if (flags.annealingType.equalsIgnoreCase("exp") || flags.annealingType.equalsIgnoreCase("exponential")) {
      sequence = sampler.findBestUsingAnnealing(model, CoolingSchedule.getExponentialSchedule(1.0, flags.annealingRate, flags.numSamples), sequence);
    } else {
      throw new RuntimeException("No annealing type specified");
    }

    //System.err.println(ArrayMath.toString(sequence));

    if (flags.useReverse) {
      Collections.reverse(document);
    }

    for (int j = 0, dsize = newDocument.size(); j < dsize; j++) {
      CoreLabel wi = document.get(j);
      if (wi==null) throw new RuntimeException("");
      if (classIndex==null) throw new RuntimeException("");
      wi.set(AnswerAnnotation.class, classIndex.get(sequence[j]));
    }

    if (flags.useReverse) {
      Collections.reverse(document);
    }

    return document;
  }

  /**
   * Takes a {@link List} of {@link CoreLabel}s and prints the likelihood
   * of each possible label at each point.
   *
   * @param document A {@link List} of {@link CoreLabel}s.
   */
  @Override
  public void printProbsDocument(List<CoreLabel> document) {

    Pair<int[][][],int[]> p = documentToDataAndLabels(document);
    int[][][] data = p.first();

    //FactorTable[] factorTables = CRFLogConditionalObjectiveFunction.getCalibratedCliqueTree(weights, data, labelIndices, classIndex.size());
    CRFCliqueTree cliqueTree = CRFCliqueTree.getCalibratedCliqueTree(weights, data, labelIndices, classIndex.size(), classIndex, flags.backgroundSymbol);

    //    for (int i = 0; i < factorTables.length; i++) {
    for (int i = 0; i < cliqueTree.length(); i++) {
      CoreLabel wi = document.get(i);
      System.out.print(wi.word() + "\t");
      for (Iterator<String> iter = classIndex.iterator(); iter.hasNext();) {
        String label = iter.next();
        int index = classIndex.indexOf(label);
        //        double prob = Math.pow(Math.E, factorTables[i].logProbEnd(index));
        double prob = cliqueTree.prob(i, index);
        System.out.print(label + "=" + prob);
        if (iter.hasNext()) {
          System.out.print("\t");
        } else {
          System.out.print("\n");
        }
      }
    }
  }

  /**
   * Takes the file, reads it in, and prints out the likelihood of
   * each possible label at each point. This gives a simple way to examine
   * the probability distributions of the CRF.  See
   * <code>getCliqueTrees()</code> for more.
   *
   * @param filename The path to the specified file
   */
  public void printFirstOrderProbs(String filename) {
    // only for the OCR data does this matter
    flags.ocrTrain = false;

    ObjectBank<List<CoreLabel>> docs = makeObjectBankFromFile(filename);
    printFirstOrderProbsDocuments(docs);
  }

  /**
   * Takes a {@link List} of documents and prints the likelihood
   * of each possible label at each point.
   *
   * @param documents A {@link List} of {@link List} of {@link CoreLabel}s.
   */
  public void printFirstOrderProbsDocuments(ObjectBank<List<CoreLabel>> documents) {
    for (List<CoreLabel> doc : documents) {
      printFirstOrderProbsDocument(doc);
      System.out.println();
    }
  }

  /**
   * Want to make arbitrary probability queries?  Then this is the method for you.
   * Given the filename, it reads it in and breaks it into documents, and then makes
   * a CRFCliqueTree for each document.  you can then ask the clique tree for marginals
   * and conditional probabilities of almost anything you want.
   */
  public List<CRFCliqueTree> getCliqueTrees(String filename) {
    // only for the OCR data does this matter
    flags.ocrTrain = false;

    List<CRFCliqueTree> cts = new ArrayList<CRFCliqueTree>();
    ObjectBank<List<CoreLabel>> docs = makeObjectBankFromFile(filename);
    for (List<CoreLabel> doc : docs) {
      cts.add(getCliqueTree(doc));
    }

    return cts;
  }


  private CRFCliqueTree getCliqueTree(List<CoreLabel> document) {

    Pair<int[][][],int[]> p = documentToDataAndLabels(document);
    int[][][] data = p.first();

    //FactorTable[] factorTables = CRFLogConditionalObjectiveFunction.getCalibratedCliqueTree(weights, data, labelIndices, classIndex.size());
    return CRFCliqueTree.getCalibratedCliqueTree(weights, data, labelIndices, classIndex.size(), classIndex, flags.backgroundSymbol);
  }

  /**
   * Takes a {@link List} of {@link CoreLabel}s and prints the likelihood
   * of each possible label at each point.
   *
   * @param document A {@link List} of {@link CoreLabel}s.
   */
  public void printFirstOrderProbsDocument(List<CoreLabel> document) {

    CRFCliqueTree cliqueTree = getCliqueTree(document);

    //    for (int i = 0; i < factorTables.length; i++) {
    for (int i = 0; i < cliqueTree.length(); i++) {
      CoreLabel wi = document.get(i);
      System.out.print(wi.word() + "\t");
      for (Iterator<String> iter = classIndex.iterator(); iter.hasNext();) {
        String label = iter.next();
        int index = classIndex.indexOf(label);
        if (i == 0) {
          //double prob = Math.pow(Math.E, factorTables[i].logProbEnd(index));
          double prob = cliqueTree.prob(i, index);
          System.out.print(label + "=" + prob);
          if (iter.hasNext()) {
            System.out.print("\t");
          } else {
            System.out.print("\n");
          }
        } else {
          for (Iterator<String> iter1 = classIndex.iterator(); iter1.hasNext();) {
            String label1 = iter1.next();
            int index1 = classIndex.indexOf(label1);
            //double prob = Math.pow(Math.E, factorTables[i].logProbEnd(new int[]{index1, index}));
            double prob = cliqueTree.prob(i, new int[]{index1, index});
            System.out.print(label1 + "_" + label + "=" + prob);
            if (iter.hasNext() || iter1.hasNext()) {
              System.out.print("\t");
            } else {
              System.out.print("\n");
            }
          }
        }
      }
    }
  }

  /** Train a classifier from documents.
   *
   *  @param docs An objectbank representation of documents.
   */
  @Override
  public void train(ObjectBank<List<CoreLabel>> docs) {
    makeAnswerArraysAndTagIndex(docs);

    for (int i = 0; i <= flags.numTimesPruneFeatures; i++) {

      Pair dataAndLabels = documentsToDataAndLabels(docs);
      if (flags.numTimesPruneFeatures == i) {
        docs = null; // hopefully saves memory
      }

      // save feature index to disk and read in later
      File featIndexFile = null;

      if (flags.saveFeatureIndexToDisk) {
        try {
          System.err.println("Writing feature index to temporary file.");
          featIndexFile = IOUtils.writeObjectToTempFile(featureIndex, "featIndex" + i+ ".tmp");
          featureIndex = null;
        } catch (IOException e) {
          throw new RuntimeException("Could not open temporary feature index file for writing.");
        }
      }

      // first index is the number of the document
      // second index is position in the document also the index of the clique/factor table
      // third index is the number of elements in the clique/window thase features are for (starting with last element)
      // fourth index is position of the feature in the array that holds them
      // element in data[i][j][k][m] is the index of the mth feature occurring in position k of the jth clique of the ith document
      int[][][][] data = (int[][][][]) dataAndLabels.first();
      // first index is the number of the document
      // second index is the position in the document
      // element in labels[i][j] is the index of the correct label (if it exists) at position j in document i
      int[][] labels = (int[][]) dataAndLabels.second();


      if (flags.loadProcessedData != null) {
        List processedData = loadProcessedData(flags.loadProcessedData);
        if (processedData != null) {
          // enlarge the data and labels array
          int[][][][] allData = new int[data.length + processedData.size()][][][];
          int[][] allLabels = new int[labels.length + processedData.size()][];
          System.arraycopy(data, 0, allData, 0, data.length);
          System.arraycopy(labels, 0, allLabels, 0, labels.length);
          // add to the data and labels array
          addProcessedData(processedData, allData, allLabels, data.length);
          data = allData;
          labels = allLabels;
        }
      }

      if (flags.useFloat) {
        CRFLogConditionalObjectiveFloatFunction func = new CRFLogConditionalObjectiveFloatFunction(data, labels, featureIndex, windowSize, classIndex, labelIndices, map, flags.backgroundSymbol, flags.sigma);
        func.crfType = flags.crfType;

        QNMinimizer minimizer;
        if (flags.interimOutputFreq != 0) {
          FloatFunction monitor = new ResultStoringFloatMonitor(flags.interimOutputFreq, flags.serializeTo);
          minimizer = new QNMinimizer(monitor);
        } else {
          minimizer = new QNMinimizer();
        }

        if (i == 0) {
          minimizer.setM(flags.QNsize);
        } else {
          minimizer.setM(flags.QNsize2);
        }

        float[] initialWeights;
        if (flags.initialWeights == null) {
          initialWeights = func.initial();
        } else {
          try {
            System.err.println("Reading initial weights from file " + flags.initialWeights);
            DataInputStream dis = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(flags.initialWeights))));
            initialWeights = Convert.readFloatArr(dis);
          } catch (IOException e) {
            throw new RuntimeException("Could not read from float initial weight file " + flags.initialWeights);
          }
        }
        System.err.println("numWeights: " + initialWeights.length);
        float[] weights = minimizer.minimize(func, (float) flags.tolerance, initialWeights);
        this.weights = ArrayMath.floatArrayToDoubleArray(func.to2D(weights));

      } else {

        /*double[] estimate = null;

        if(flags.estimateInitial){
          int[][][][] approxData = new int[data.length/100][][][];
          int[][] approxLabels = new int[data.length/100][];

          Random generator = new Random(1);
          for(int k=0;k<approxData.length; k++){
            int thisInd = generator.nextInt(data.length);
            approxData[k] = data[ thisInd];
            approxLabels[k] = labels[ thisInd];
          }

          CRFLogConditionalObjectiveFunction approxFunc = new CRFLogConditionalObjectiveFunction(approxData, approxLabels, featureIndex, windowSize, classIndex, labelIndices, map, flags.backgroundSymbol, flags.sigma);
          approxFunc.crfType = flags.crfType;
          minimizer = new QNMinimizer(10);

          if (flags.initialWeights == null) {
            estimate = approxFunc.initial();
          } else {
            try {
              System.err.println("Reading initial weights from file " + flags.initialWeights);
              DataInputStream dis = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(flags.initialWeights))));
              estimate = Convert.readDoubleArr(dis);
            } catch (IOException e) {
              throw new RuntimeException("Could not read from double initial weight file " + flags.initialWeights);
            }
          }

          estimate = minimizer.minimize(approxFunc, 1e-2, estimate);

        }
        */

        CRFLogConditionalObjectiveFunction func = new CRFLogConditionalObjectiveFunction(data, labels, featureIndex, windowSize, classIndex, labelIndices, map, flags.backgroundSymbol, flags.sigma);

        func.crfType = flags.crfType;

        minimizer = getMinimizer(i);

        double[] initialWeights;
        if (flags.initialWeights == null) {
          initialWeights = func.initial();
        } else {
          try {
            System.err.println("Reading initial weights from file " + flags.initialWeights);
            DataInputStream dis = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(flags.initialWeights))));
            initialWeights = Convert.readDoubleArr(dis);
          } catch (IOException e) {
            throw new RuntimeException("Could not read from double initial weight file " + flags.initialWeights);
          }
        }
        System.err.println("numWeights: " + initialWeights.length);

        if (flags.testObjFunction) {
          StochasticDiffFunctionTester tester = new StochasticDiffFunctionTester(func);
          if(tester.testSumOfBatches(initialWeights,1e-4)){
            System.err.println("Testing complete... exiting");
            System.exit(1);
          } else {
            System.err.println("Testing failed....exiting");
            System.exit(1);
          }

        }
        double[] weights = minimizer.minimize(func, flags.tolerance, initialWeights);
        this.weights = func.to2D(weights);
      }

      // save feature index to disk and read in later
      if (flags.saveFeatureIndexToDisk) {
        try {
          System.err.println("Reading temporary feature index file.");
          featureIndex = (Index<String>) IOUtils.readObjectFromFile(featIndexFile);
        } catch (Exception e) {
          throw new RuntimeException("Could not open temporary feature index file for reading.");
        }
      }

      if (i != flags.numTimesPruneFeatures) {
        dropFeaturesBelowThreshold(flags.featureDiffThresh);
        System.err.println("Removing features with weight below " + flags.featureDiffThresh + " and retraining...");
      }

    }
  }

  protected Minimizer getMinimizer(){
    return getMinimizer(0);
  }

  protected Minimizer getMinimizer(int featurePruneIteration){
    if( flags.useQN ){

      int QNmem;
      if (featurePruneIteration == 0) {
        QNmem = flags.QNsize;
      } else {
        QNmem = flags.QNsize2;
      }

      if (flags.interimOutputFreq != 0) {
        Function monitor = new ResultStoringMonitor(flags.interimOutputFreq, flags.serializeTo);
        minimizer = new QNMinimizer(monitor,QNmem,flags.useRobustQN);
      } else {
        minimizer = new QNMinimizer(QNmem,flags.useRobustQN);
      }

    } else if( flags.useSGDtoQN ) {
      minimizer = new SGDToQNMinimizer(flags);
    } else if( flags.useSMD){
      minimizer = new SMDMinimizer(flags.initialGain, flags.stochasticBatchSize, flags.stochasticMethod,flags.SGDPasses);
    } else if( flags.useSGD){
      minimizer = new SGDMinimizer(flags.initialGain,flags.stochasticBatchSize);
    } else if( flags.useScaledSGD){
      minimizer = new ScaledSGDMinimizer(flags.initialGain,flags.stochasticBatchSize,flags.SGDPasses,flags.scaledSGDMethod);
    }

    if(minimizer==null){
      throw new RuntimeException("No minimizer assigned!");
    }

    return minimizer;
  }

  /**
   * Creates a new CRFDatum from the preprocessed allData format, given the document number,
   * position number, and a List of Object labels.
   *
   * @param allData
   * @param beginPosition
   * @param endPosition
   * @param labeledWordInfos
   * @return A new CRFDatum
   */
  protected List<CRFDatum> extractDatumSequence(int[][][] allData, int beginPosition, int endPosition, List<CoreLabel> labeledWordInfos) {
    List<CRFDatum> result = new ArrayList<CRFDatum>();
    int beginContext = beginPosition - windowSize + 1;
    if (beginContext < 0) {
      beginContext = 0;
    }
    // for the beginning context, add some dummy datums with no features!
    // TODO: is there any better way to do this?
    for (int position = beginContext; position < beginPosition; position++) {
      List cliqueFeatures = new ArrayList();
      for (int i = 0; i < windowSize; i++) {
        // create a feature list
        cliqueFeatures.add(Collections.EMPTY_SET);
      }
      CRFDatum<Serializable,String> datum = new CRFDatum<Serializable,String>(cliqueFeatures, labeledWordInfos.get(position).get(AnswerAnnotation.class));
      result.add(datum);
    }
    // now add the real datums
    for (int position = beginPosition; position <= endPosition; position++) {
      List cliqueFeatures = new ArrayList();
      for (int i = 0; i < windowSize; i++) {
        // create a feature list
        Collection<String> features = new ArrayList<String>();
        for (int j = 0; j < allData[position][i].length; j++) {
          features.add(featureIndex.get(allData[position][i][j]));
        }
        cliqueFeatures.add(features);
      }
      CRFDatum<Serializable,String> datum = new CRFDatum<Serializable,String>(cliqueFeatures, labeledWordInfos.get(position).get(AnswerAnnotation.class));
      result.add(datum);
    }
    return result;
  }

  /**
   * Adds the List of Lists of CRFDatums to the data and labels arrays, treating each datum as if
   * it were its own document.
   * Adds context labels in addition to the target label for each datum, meaning that for a particular
   * document, the number of labels will be windowSize-1 greater than the number of datums.
   *
   * @param processedData a List of Lists of CRFDatums
   * @param data
   * @param labels
   * @param offset
   */
  protected void addProcessedData(List<List<CRFDatum>> processedData, int[][][][] data, int[][] labels, int offset) {
    for (int i = 0, pdSize = processedData.size(); i < pdSize; i++) {
      int dataIndex = i + offset;
      List<CRFDatum> document = processedData.get(i);
      int dsize = document.size();
      labels[dataIndex] = new int[dsize];
      data[dataIndex] = new int[dsize][][];
      for (int j = 0; j < dsize; j++) {
        CRFDatum crfDatum = document.get(j);
        // add label, they are offset by extra context
        labels[dataIndex][j] = classIndex.indexOf((String) crfDatum.label());
        // add features
        List<Collection<String>> cliques = crfDatum.asFeatures();
        int csize = cliques.size();
        data[dataIndex][j] = new int[csize][];
        for (int k = 0; k < csize; k++) {
          Collection<String> features = cliques.get(k);

          // Debug only: Remove
          // if (j < windowSize) {
          //   System.err.println("addProcessedData: Features Size: " + features.size());
          // }

          data[dataIndex][j][k] = new int[features.size()];

          int m = 0;
          try {
            for (String feature : features) {
              //System.err.println("feature " + feature);
              //              if (featureIndex.indexOf(feature)) ;
              if (featureIndex == null) {
                System.out.println("Feature is NULL!");
              }
              data[dataIndex][j][k][m] = featureIndex.indexOf(feature);
              m++;
            }
          } catch (Exception e) {
            e.printStackTrace();
            System.err.printf("[index=%d, j=%d, k=%d, m=%d]\n", dataIndex, j, k, m);
            System.err.println("data.length                    " + data.length);
            System.err.println("data[dataIndex].length         " + data[dataIndex].length);
            System.err.println("data[dataIndex][j].length      " + data[dataIndex][j].length);
            System.err.println("data[dataIndex][j][k].length   " + data[dataIndex][j].length);
            System.err.println("data[dataIndex][j][k][m]       " + data[dataIndex][j][k][m]);
            return;
          }
        }
      }
    }
  }

  protected static void saveProcessedData(List datums, String filename) {
    System.err.print("Saving processsed data of size " + datums.size() + " to serialized file...");
    ObjectOutputStream oos = null;
    try {
      oos = new ObjectOutputStream(new FileOutputStream(filename));
      oos.writeObject(datums);
    } catch (IOException e) {
      // do nothing
    } finally {
      if (oos != null) {
        try {
          oos.close();
        } catch (IOException e) {
        }
      }
    }
    System.err.println("done.");
  }

  protected static List loadProcessedData(String filename) {
    System.err.print("Loading processed data from serialized file...");
    ObjectInputStream ois = null;
    List result = Collections.EMPTY_LIST;
    try {
      ois = new ObjectInputStream(new FileInputStream(filename));
      result = (List) ois.readObject();
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      if (ois != null) {
        try {
          ois.close();
        } catch (IOException e) {
        }
      }
    }
    System.err.println("done. Got " + result.size() + " datums.");
    return result;
  }

  public void loadTextClassifier(String text, Properties props) throws ClassCastException, IOException, ClassNotFoundException, InstantiationException, IllegalAccessException {
    //System.err.println("DEBUG: in loadTextClassifier");
    System.err.println("Loading Text Classifier from "+text);
    BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(text))));

    String line = br.readLine();
    // first line should be this format:
    // labelIndices.length=\t%d
    String[] toks = line.split("\\t");
    if (!toks[0].equals("labelIndices.length=")) { throw new RuntimeException("format error"); }
    int size = Integer.parseInt(toks[1]);
    labelIndices = new Index[size];
    for (int labelIndicesIdx = 0; labelIndicesIdx < size; labelIndicesIdx++) {
      line = br.readLine();
      // first line should be this format:
      // labelIndices.length=\t%d
      // labelIndices[0].size()=\t%d
      toks = line.split("\\t");
      if (! (toks[0].startsWith("labelIndices[") && toks[0].endsWith("].size()="))) {
        throw new RuntimeException("format error");
      }
      int labelIndexSize = Integer.parseInt(toks[1]);
      labelIndices[labelIndicesIdx] = new Index<CRFLabel>();
      int count = 0;
      while(count<labelIndexSize) {
        line = br.readLine();
        toks = line.split("\\t");
        int idx = Integer.parseInt(toks[0]);
        if (count!=idx) { throw new RuntimeException("format error"); }

        String[] crflabelstr = toks[1].split(" ");
        int[] crflabel = new int[crflabelstr.length];
        for (int i=0; i < crflabelstr.length; i++) {
          crflabel[i] = Integer.parseInt(crflabelstr[i]);
        }
        CRFLabel crfL = new CRFLabel(crflabel);

        labelIndices[labelIndicesIdx].add(crfL);
        count++;
      }
    }

    /**************************************/
    System.err.printf("DEBUG: labelIndices.length=\t%d\n",labelIndices.length);
    for(int i = 0; i < labelIndices.length; i++) {
      System.err.printf("DEBUG: labelIndices[%d].size()=\t%d\n", i, labelIndices[i].size());
      for(int j = 0; j < labelIndices[i].size(); j++) {
        int[] label = labelIndices[i].get(j).getLabel();
        List<Integer> list = new ArrayList<Integer>();
        for(int l : label) {
          list.add(l);
        }
        System.err.printf("DEBUG: %d\t%s\n", j, StringUtils.join(list, " "));
      }
    }
    /**************************************/


    line = br.readLine();
    toks = line.split("\\t");
    if (!toks[0].equals("classIndex.size()=")) { throw new RuntimeException("format error"); }
    int classIndexSize = Integer.parseInt(toks[1]);
    classIndex = new Index<String>();
    int count = 0;
    while(count<classIndexSize) {
      line = br.readLine();
      toks = line.split("\\t");
      int idx = Integer.parseInt(toks[0]);
      if (count!=idx) { throw new RuntimeException("format error"); }
      classIndex.add(toks[1]);
      count++;
    }

    /******************************************/
    System.err.printf("DEBUG: classIndex.size()=\t%d\n", classIndex.size());
    for(int i = 0; i < classIndex.size(); i++) {
      System.err.printf("DEBUG: %d\t%s\n", i, classIndex.get(i));
    }
    /******************************************/


    line = br.readLine();
    toks = line.split("\\t");
    if (!toks[0].equals("featureIndex.size()=")) { throw new RuntimeException("format error"); }
    int featureIndexSize = Integer.parseInt(toks[1]);
    featureIndex = new Index<String>();
    count = 0;
    while(count<featureIndexSize) {
      line = br.readLine();
      toks = line.split("\\t");
      int idx = Integer.parseInt(toks[0]);
      if (count!=idx) { throw new RuntimeException("format error"); }
      featureIndex.add(toks[1]);
      count++;
    }

    /***************************************/
    System.err.printf("DEBUG: featureIndex.size()=\t%d\n", featureIndex.size());
    /*
      for(int i = 0; i < featureIndex.size(); i++) {
        System.err.printf("DEBUG: %d\t%s\n", i, featureIndex.get(i));
      }
    */
    /***************************************/


    line = br.readLine();
    if (!line.equals("<flags>")) { throw new RuntimeException("format error"); }
    Properties p = new Properties();
    line = br.readLine();

    while(!line.equals("</flags>")) {
      //System.err.println("DEBUG: flags line: "+line);
      String[] keyValue = line.split("=");
      //System.err.printf("DEBUG: p.setProperty(%s,%s)\n", keyValue[0], keyValue[1]);
      p.setProperty(keyValue[0], keyValue[1]);
      line = br.readLine();
    }

    //System.err.println("DEBUG: out from flags");
    flags = new SeqClassifierFlags(p);
    System.err.println("DEBUG: <flags>");
    System.err.print(flags.toString());
    System.err.println("DEBUG: </flags>");

    // <featureFactory> edu.stanford.nlp.wordseg.Gale2007ChineseSegmenterFeatureFactory </featureFactory>
    line = br.readLine();

    String[] featureFactoryName = line.split(" ");
    if (!featureFactoryName[0].equals("<featureFactory>")
        || !featureFactoryName[2].equals("</featureFactory>")) {
      throw new RuntimeException("format error");
    }
    featureFactory = (edu.stanford.nlp.sequences.FeatureFactory)Class.forName(featureFactoryName[1]).newInstance();
    featureFactory.init(flags);


    reinit();

    // <windowSize> 2 </windowSize>
    line = br.readLine();

    String[] windowSizeName = line.split(" ");
    if (!windowSizeName[0].equals("<windowSize>")
        || !windowSizeName[2].equals("</windowSize>")) {
      throw new RuntimeException("format error");
    }
    windowSize = Integer.parseInt(windowSizeName[1]);

    // weights.length= 2655170
    line = br.readLine();

    toks = line.split("\\t");
    if (!toks[0].equals("weights.length=")) { throw new RuntimeException("format error"); }
    int weightsLength = Integer.parseInt(toks[1]);
    weights = new double[weightsLength][];
    count = 0;
    while(count < weightsLength) {
      line = br.readLine();

      toks = line.split("\\t");
      int weights2Length = Integer.parseInt(toks[0]);
      weights[count] = new double[weights2Length];
      String[] weightsValue = toks[1].split(" ");
      if (weights2Length != weightsValue.length)
      { throw new RuntimeException("weights format error"); }

      for(int i2 = 0; i2 < weights2Length; i2++) {
        weights[count][i2] = Double.parseDouble(weightsValue[i2]);
      }
      count++;
    }
    System.err.printf("DEBUG: double[%d][] weights loaded\n", weightsLength);
    line = br.readLine();

    if (line != null)
    { throw new RuntimeException("weights format error"); }
  }

  /**
   * Serialize the model to a human readable format.
   * It's not yet complete. It should now work for Chinese segmenter though.
   * TODO: check things in serializeClassifier and add other necessary serialization back
   *
   * @param serializePath File to write text format of classifier to.
   */
  public void serializeTextClassifier(String serializePath) {
    System.err.print("Serializing Text classifier to " + serializePath + "...");
    try {
      PrintWriter pw = new PrintWriter(new GZIPOutputStream(new FileOutputStream(serializePath)));

      pw.printf("labelIndices.length=\t%d\n",labelIndices.length);
      for(int i = 0; i < labelIndices.length; i++) {
        pw.printf("labelIndices[%d].size()=\t%d\n", i, labelIndices[i].size());
        for(int j = 0; j < labelIndices[i].size(); j++) {
          int[] label = labelIndices[i].get(j).getLabel();
          List<Integer> list = new ArrayList<Integer>();
          for(int l : label) {
            list.add(l);
          }
          pw.printf("%d\t%s\n", j, StringUtils.join(list, " "));
        }
      }

      pw.printf("classIndex.size()=\t%d\n", classIndex.size());
      for(int i = 0; i < classIndex.size(); i++) {
        pw.printf("%d\t%s\n", i, classIndex.get(i));
      }
      //pw.printf("</classIndex>\n");

      pw.printf("featureIndex.size()=\t%d\n", featureIndex.size());
      for(int i = 0; i < featureIndex.size(); i++) {
        pw.printf("%d\t%s\n", i, featureIndex.get(i));
      }
      //pw.printf("</featureIndex>\n");

      pw.println("<flags>");
      pw.print(flags.toString());
      pw.println("</flags>");

      pw.printf("<featureFactory> %s </featureFactory>\n",featureFactory.getClass().getName());

      pw.printf("<windowSize> %d </windowSize>\n", windowSize);

      pw.printf("weights.length=\t%d\n", weights.length);
      for (double[] ws : weights) {
        ArrayList<Double> list = new ArrayList<Double>();
        for (double w : ws) {
          list.add(w);
        }
        pw.printf("%d\t%s\n", ws.length, StringUtils.join(list, " "));
      }

      pw.close();
      System.err.println("done.");

    } catch (Exception e) {
      System.err.println("Failed");
      e.printStackTrace();
      // don't actually exit in case they're testing too
      //System.exit(1);
    }
  }


  /** {@inheritDoc}
   */
  @Override
  public void serializeClassifier(String serializePath) {
    System.err.print("Serializing classifier to " + serializePath + "...");

    try {
      ObjectOutputStream oos = IOUtils.writeStreamFromString(serializePath);

      oos.writeObject(labelIndices);
      oos.writeObject(classIndex);
      oos.writeObject(featureIndex);
      oos.writeObject(flags);
      oos.writeObject(featureFactory);
      oos.writeInt(windowSize);
      oos.writeObject(weights);
      //oos.writeObject(WordShapeClassifier.getKnownLowerCaseWords());
      if (readerAndWriter instanceof TrueCasingDocumentReaderAndWriter) {
        oos.writeObject(TrueCasingDocumentReaderAndWriter.knownWords);
      }

      oos.writeObject(knownLCWords);

      oos.close();
      System.err.println("done.");

    } catch (Exception e) {
      System.err.println("Failed");
      e.printStackTrace();
      // don't actually exit in case they're testing too
      //System.exit(1);
    }
  }

  
  /**
   * Loads a classifier from the specified InputStream.
   * This version works quietly (unless VERBOSE is true).
   * If props is non-null then any properties it specifies override
   * those in the serialized file.  However, only some properties are
   * sensible to change (you shouldn't change how features are defined).
   * <p>
   * <i>Note:</i> This method does not close the ObjectInputStream.  (But
   * earlier versions of the code used to, so beware....)
   */
  @Override
  @SuppressWarnings({"unchecked"}) // can't have right types in deserialization
  public void loadClassifier(ObjectInputStream ois, Properties props) throws ClassCastException, IOException, ClassNotFoundException {
    labelIndices = (Index<CRFLabel>[]) ois.readObject();
    classIndex = (Index<String>) ois.readObject();
    featureIndex = (Index<String>) ois.readObject();
    flags = (SeqClassifierFlags) ois.readObject();
    featureFactory = (edu.stanford.nlp.sequences.FeatureFactory) ois.readObject();

    if (props != null) {
      flags.setProperties(props, false);
    }
    reinit();

    windowSize = ois.readInt();
    weights = (double[][]) ois.readObject();

    if (readerAndWriter instanceof TrueCasingDocumentReaderAndWriter) {
      TrueCasingDocumentReaderAndWriter.knownWords = (Set)ois.readObject();
    }

    //WordShapeClassifier.setKnownLowerCaseWords((Set) ois.readObject());
    knownLCWords = (Set<String>) ois.readObject();

    if (VERBOSE) {
      System.err.println("windowSize=" + windowSize);
      System.err.println("flags=\n" + flags);
    }
  }

  /**
   * This is used to load the default supplied classifier stored within
   * the jar file.
   * THIS FUNCTION WILL ONLY WORK IF THE CODE WAS LOADED FROM A JAR FILE
   * WHICH HAS A SERIALIZED CLASSIFIER STORED INSIDE IT.
   */
  public void loadDefaultClassifier() {
    loadJarClassifier(DEFAULT_CLASSIFIER, null);
  }


  /**
   * Used to get the default supplied classifier inside the jar file.
   * THIS FUNCTION WILL ONLY WORK IF THE CODE WAS LOADED FROM A JAR FILE
   * WHICH HAS A SERIALIZED CLASSIFIER STORED INSIDE IT.
   *
   * @return The default CRFClassifier in the jar file (if there is one)
   */
  public static CRFClassifier getDefaultClassifier() {
    CRFClassifier crf = new CRFClassifier();
    crf.loadDefaultClassifier();
    return crf;
  }

  /**
   * Used to load a classifier stored as a resource inside a jar file.
   * THIS FUNCTION WILL ONLY WORK IF THE CODE WAS LOADED FROM A JAR FILE
   * WHICH HAS A SERIALIZED CLASSIFIER STORED INSIDE IT.
   *
   * @param resourceName Name of clasifier resource inside the jar file.
   * @return A CRFClassifier stored in the jar file
   */
  public static CRFClassifier getJarClassifier(String resourceName, Properties props) {
    CRFClassifier crf = new CRFClassifier();
    crf.loadJarClassifier(resourceName, props);
    return crf;
  }


  /** Loads a CRF classifier from a filepath, and returns it.
   *
   *  @param file File to load classifier from
   *  @return The CRF classifier
   *
   *  @throws IOException If there are problems accessing the input stream
   *  @throws ClassCastException If there are problems interpreting the serialized data
   *  @throws ClassNotFoundException If there are problems interpreting the serialized data
   */
  public static CRFClassifier getClassifier(File file) throws IOException, ClassCastException, ClassNotFoundException {
    CRFClassifier crf = new CRFClassifier();
    crf.loadClassifier(file);
    return crf;
  }

  /** Loads a CRF classifier from an InputStream, and returns it.  This method
   *  does not buffer the InputStream, so you should have buffered it before
   *  calling this method.
   *
   *  @param in InputStream to load classifier from
   *  @return The CRF classifier
   *
   *  @throws IOException If there are problems accessing the input stream
   *  @throws ClassCastException If there are problems interpreting the serialized data
   *  @throws ClassNotFoundException If there are problems interpreting the serialized data
   */
  public static CRFClassifier getClassifier(InputStream in) throws IOException, ClassCastException, ClassNotFoundException {
    CRFClassifier crf = new CRFClassifier();
    crf.loadClassifier(in);
    return crf;
  }

  public static CRFClassifier getClassifierNoExceptions(String loadPath) {
    CRFClassifier crf = new CRFClassifier();
    crf.loadClassifierNoExceptions(loadPath);
    return crf;
  }

  public static CRFClassifier getClassifier(String loadPath) throws IOException, ClassCastException, ClassNotFoundException {
    CRFClassifier crf = new CRFClassifier();
    crf.loadClassifier(loadPath);
    return crf;
  }

  /** The main method. See the class documentation. */
  public static void main(String[] args) throws Exception {
    StringUtils.printErrInvocationString("CRFClassifier", args);

    Properties props = StringUtils.argsToProperties(args);
    CRFClassifier crf = new CRFClassifier(props);
    String testFile = crf.flags.testFile;
    String textFile = crf.flags.textFile;
    String loadPath = crf.flags.loadClassifier;
    String loadTextPath = crf.flags.loadTextClassifier;
    String serializeTo = crf.flags.serializeTo;
    String serializeToText = crf.flags.serializeToText;

    if (loadPath != null) {
      crf.loadClassifierNoExceptions(loadPath, props);
    } else if (loadTextPath != null) {
      System.err.println("Warning: this is now only tested for Chinese Segmenter");
      System.err.println("(Sun Dec 23 00:59:39 2007) (pichuan)");
      try {
        crf.loadTextClassifier(loadTextPath, props);
        //System.err.println("DEBUG: out from crf.loadTextClassifier");
      } catch (Exception e) {
        e.printStackTrace();
        throw new RuntimeException("error loading "+loadTextPath);
      }
    } else if (crf.flags.loadJarClassifier != null) {
      crf.loadJarClassifier(crf.flags.loadJarClassifier, props);
    } else if (crf.flags.trainFile != null || crf.flags.trainFileList != null) {
      crf.train();
    } else {
      crf.loadDefaultClassifier();
    }

    // System.err.println("Using " + crf.flags.featureFactory);
    // System.err.println("Using " + StringUtils.getShortClassName(crf.readerAndWriter));

    if (serializeTo != null) {
      crf.serializeClassifier(serializeTo);
    }

    if (serializeToText != null) {
      crf.serializeTextClassifier(serializeToText);
    }

    if (testFile != null) {
      if (crf.flags.searchGraphPrefix != null) {
        crf.classifyAndWriteViterbiSearchGraph(testFile,crf.flags.searchGraphPrefix);
      } else if (crf.flags.printFirstOrderProbs) {
        crf.printFirstOrderProbs(testFile);
      } else if (crf.flags.printProbs) {
        crf.printProbs(testFile);
      } else if (crf.flags.useKBest) {
        int k = crf.flags.kBest;
        crf.classifyAndWriteAnswersKBest(testFile, k);
      } else if (crf.flags.printLabelValue) {
        crf.printLabelInformation(testFile);
      } else {
        crf.classifyAndWriteAnswers(testFile);
      }
    }

    if (textFile != null) {
      DocumentReaderAndWriter oldRW = crf.readerAndWriter;
      crf.readerAndWriter = new PlainTextDocumentReaderAndWriter();
      crf.readerAndWriter.init(crf.flags);
      crf.classifyAndWriteAnswers(textFile);
      crf.readerAndWriter = oldRW;
    }
  } // end main

} // end class CRFClassifier
