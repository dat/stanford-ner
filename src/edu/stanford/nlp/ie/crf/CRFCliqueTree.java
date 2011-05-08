package edu.stanford.nlp.ie.crf;

import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.sequences.SequenceListener;
import edu.stanford.nlp.sequences.SequenceModel;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.GeneralizedCounter;
import edu.stanford.nlp.util.Index;

import java.util.Arrays;
import java.util.List;


/** Builds a CliqueTree (an array of FactorTable) and does message passing
 *  inference along it.
 *
 *  @author Jenny Finkel
 */
public class CRFCliqueTree implements SequenceModel, SequenceListener {
  
  private FactorTable[] factorTables;
  private double z; // norm constant
  private Index classIndex;
  private String backgroundSymbol;
  private int backgroundIndex;
  // the window size, which is also the clique size
  private int windowSize;
  // the number of possible classes for each label
  private int numClasses;
  private int[] possibleValues;

  private CRFCliqueTree() {}

  private CRFCliqueTree(FactorTable[] factorTables, Index classIndex, String backgroundSymbol) {
    this.factorTables = factorTables;
    this.classIndex = classIndex;
    this.backgroundSymbol = backgroundSymbol;
    backgroundIndex = classIndex.indexOf(backgroundSymbol);
    z = factorTables[0].totalMass();
    windowSize = factorTables[0].windowSize();
    numClasses = classIndex.size();
    possibleValues = new int[numClasses];
    for (int i=0; i<numClasses; i++) {
      possibleValues[i] = i;
    }

    //Debug only
    //System.out.println("CRFCliqueTree constructed::numClasses: " + numClasses);
  }

  public Index classIndex() {
    return classIndex;
  }

  // SEQUENCE MODEL METHODS

  public int length() {
    return factorTables.length;
  }

  public int leftWindow() {
    return windowSize;
  }

  public int rightWindow() {
    return 0;
  }

  public int[] getPossibleValues(int position)  {
    return possibleValues;
  }

  public double scoreOf(int[] sequence, int pos) {
    return scoresOf(sequence, pos)[sequence[pos]];
  }

  /**
   * Computes the unnormalized log conditional distribution over values of the
   * element at position pos in the sequence, conditioned on the values
   *  of the elements in all other positions of the provided sequence.
   *
   * @param sequence the sequence containing the rest of the values to condition on
   * @param position     the position of the element to give a distribution for
   * @return an array of type double, representing a probability distribution; sums to 1.0
   */
  public double[] scoresOf(int[] sequence, int position) {
    if (position>=factorTables.length) throw new RuntimeException("Index out of bounds: " + position);
//    DecimalFormat nf = new DecimalFormat("#0.000");
    // if (position>0 && position<sequence.length-1) System.out.println(position + ": asking about " +sequence[position-1] + "(" + sequence[position] + ")" + sequence[position+1]);
    double[] probThisGivenPrev = new double[numClasses];
    double[] probNextGivenThis = new double[numClasses];
//    double[] marginal = new double[numClasses];     // for debugging only

    // compute prob of this tag given the window-1 previous tags, normalized
    // extract the window-1 previous tags, pad left with background if necessary
    int prevLength = windowSize-1;
    int[] prev = new int[prevLength+1]; // leave an extra element for the label at this position
    int i=0;
    for ( ; i<prevLength-position; i++) { // will only happen if position-prevLength < 0
      prev[i] = classIndex.indexOf(backgroundSymbol);
    }
    for ( ; i<prevLength; i++) {
      prev[i] = sequence[position-prevLength+i];
    }
    for (int label=0; label<numClasses; label++) {
      prev[prev.length-1] = label;
      probThisGivenPrev[label] = factorTables[position].unnormalizedLogProb(prev);
//      marginal[label] = factorTables[position].logProbEnd(label);     // remove: for debugging only
    }

    // ArrayMath.logNormalize(probThisGivenPrev);

    // compute the prob of the window-1 next tags given this tag
    // extract the window-1 next tags
    int nextLength = windowSize-1;
    if (position+nextLength>=length()) {
      nextLength = length()-position-1;
    }
    FactorTable nextFactorTable = factorTables[position+nextLength];
    if (nextLength != windowSize - 1) {
      for (int j = 0; j < windowSize - 1 - nextLength; j++) {
        nextFactorTable = nextFactorTable.sumOutFront();
      }
    }
    if (nextLength==0) { // we are asking about the prob of no sequence
      Arrays.fill(probNextGivenThis, 1.0);
    } else {
      int[] next = new int[nextLength];
      System.arraycopy(sequence, position+1, next, 0, nextLength);
      for (int label=0; label<numClasses; label++) {
        // ask the factor table such that pos is the first position in the window
        //probNextGivenThis[label] = factorTables[position+nextLength].conditionalLogProbGivenFirst(label, next);
        //probNextGivenThis[label] = nextFactorTable.conditionalLogProbGivenFirst(label, next);
        probNextGivenThis[label] = nextFactorTable.unnormalizedConditionalLogProbGivenFirst(label, next);
      }
    }

    // pointwise multiply
    return ArrayMath.pairwiseAdd(probThisGivenPrev, probNextGivenThis);
  }



  /**
   * Returns the log probability of this sequence given the CRF.
   * Does so by computing the marginal of the first windowSize tags, and
   * then computing the conditional probability for the rest of them, conditioned on the previous
   * tags.
   * @param sequence the sequence to compute a score for
   * @return the score for the sequence
   */
  public double scoreOf(int[] sequence) {

    int[] given = new int[window() - 1];
    Arrays.fill(given, classIndex.indexOf(backgroundSymbol));
    double logProb = 0;
    for (int i=0; i<length(); i++) {
      int label = sequence[i];
      logProb += condLogProbGivenPrevious(i, label, given);
      System.arraycopy(given, 1, given, 0, given.length - 1);
      given[given.length - 1] = label;
    }
    return logProb;
  }



  // OTHER

  public int window() {
    return windowSize;
  }

  public int getNumClasses() {
    return numClasses;
  }

  public double totalMass() {
    return z;
  }

  public int backgroundIndex() {
    return backgroundIndex;
  }

  public String backgroundSymbol() {
    return backgroundSymbol;
  }

  //
  // MARGINAL PROB OF TAG AT SINGLE POSITION
  //

  public double logProb(int position, int label) {
    double u = factorTables[position].unnormalizedLogProbEnd(label);
    return u - z;
  }

  public double prob(int position, int label) {
    return Math.exp(logProb(position, label));
  }

  public double logProb(int position, Object label) {
    return logProb(position, classIndex.indexOf(label));
  }

  public double prob(int position, Object label) {
    return Math.exp(logProb(position, label));
  }

  public ClassicCounter probs(int position) {
    ClassicCounter c = new ClassicCounter();
    for (int i = 0; i < classIndex.size(); i++) {
      Object label = classIndex.get(i);
      c.incrementCount(label, prob(position, i));
    }
    return c;
  }

  public ClassicCounter logProbs(int position) {
    ClassicCounter c = new ClassicCounter();
    for (int i = 0; i < classIndex.size(); i++) {
      Object label = classIndex.get(i);
      c.incrementCount(label, logProb(position, i));
    }
    return c;
  }

  //
  // MARGINAL PROBS OF TAGS AT MULTIPLE POSITIONS
  //

  /**
   * returns the log probability for the given labels (indexed using classIndex), where the
   * last label corresponds to the label at the specified position.  For instance if you called
   * logProb(5, {1,2,3}) it will return the marginal log prob that the label at position
   * 3 is 1, the label at position 4 is 2 and the label at position 5 is 3.
   */
  public double logProb(int position, int[] labels) {
    if (labels.length < windowSize) {
      return factorTables[position].unnormalizedLogProbEnd(labels) - z;
    } else if (labels.length == windowSize) {
      return factorTables[position].unnormalizedLogProb(labels) - z;
    } else {
      int[] l = new int[windowSize];
      System.arraycopy(labels, 0, l, 0, l.length);
      int position1 = position-labels.length+windowSize;
      double p = factorTables[position1].unnormalizedLogProb(l) - z;
      l = new int[windowSize-1];
      System.arraycopy(labels, 1, l, 0, l.length);
      position1++;
      for (int i = windowSize; i < labels.length; i++) {
        p += condLogProbGivenPrevious(position1++, labels[i], l);
        System.arraycopy(l, 1, l, 0, l.length-1);
        l[windowSize-2] = labels[i];
      }      
      return p;
    }
  }

  /**
   * returns theprobability for the given labels (indexed using classIndex), where the
   * last label corresponds to the label at the specified position.  For instance if you called
   * prob(5, {1,2,3}) it will return the marginal prob that the label at position
   * 3 is 1, the label at position 4 is 2 and the label at position 5 is 3.
   */
  public double prob(int position, int[] labels) {
    return Math.exp(logProb(position, labels));
  }

  /**
   * returns the log probability for the given labels, where the
   * last label corresponds to the label at the specified position.  For instance if you called
   * logProb(5, {"O", "PER", "ORG"}) it will return the marginal log prob that the label at position
   * 3 is "O", the label at position 4 is "PER" and the label at position 5 is "ORG".
   */
  public double logProb(int position, Object[] labels) {
    return logProb(position, objectArrayToIntArray(labels));
  }

  /**
   * returns the probability for the given labels, where the
   * last label corresponds to the label at the specified position.  For instance if you called
   * logProb(5, {"O", "PER", "ORG"}) it will return the marginal prob that the label at position
   * 3 is "O", the label at position 4 is "PER" and the label at position 5 is "ORG".
   */
  public double prob(int position, Object[] labels) {
    return Math.exp(logProb(position, labels));
  }

  public GeneralizedCounter logProbs(int position, int window) {
    GeneralizedCounter gc = new GeneralizedCounter(window);
    int[] labels = new int[window];
    // cdm july 2005: below array initialization isn't necessary: JLS (3rd ed.) 4.12.5 
    // Arrays.fill(labels, 0);

    OUTER: while (true) {
      List labelsList = Arrays.asList(intArrayToObjectArray(labels));
      gc.incrementCount(labelsList, logProb(position, labels));
      for (int i = 0; i < labels.length; i++) {
        labels[i]++;
        if (labels[i] < numClasses) { break; }
        if (i == labels.length - 1) { break OUTER; }
        labels[i] = 0;
      }
    }
    return gc;
  }

  public GeneralizedCounter probs(int position, int window) {
    GeneralizedCounter gc = new GeneralizedCounter(window);
    int[] labels = new int[window];
    // cdm july 2005: below array initialization isn't necessary: JLS (3rd ed.) 4.12.5 
    // Arrays.fill(labels, 0);

    OUTER: while (true) {
      List labelsList = Arrays.asList(intArrayToObjectArray(labels));
      gc.incrementCount(labelsList, prob(position, labels));
      for (int i = 0; i < labels.length; i++) {
        labels[i]++;
        if (labels[i] < numClasses) { break; }
        if (i == labels.length - 1) { break OUTER; }
        labels[i] = 0;
      }
    }
    return gc;
  }

  //
  // HELPER METHODS
  //

  private int[] objectArrayToIntArray(Object[] os) {
    int[] is = new int[os.length];
    for (int i = 0; i < os.length; i++) {
      is[i] = classIndex.indexOf(os[i]);
    }
    return is;
  }

  private Object[] intArrayToObjectArray(int[] is) {
    Object[] os = new Object[is.length];
    for (int i = 0; i < is.length; i++) {
      os[i] = classIndex.get(is[i]);
    }
    return os;
  }

  //
  // PROB OF TAG AT SINGLE POSITION CONDITIONED ON PREVIOUS SEQUENCE OF LABELS
  //

  public double condLogProbGivenPrevious(int position, int label, int[] prevLabels) {
    if (prevLabels.length + 1 == windowSize) {
      return factorTables[position].conditionalLogProbGivenPrevious(prevLabels, label);
    } else if (prevLabels.length + 1 < windowSize) {
      FactorTable ft = factorTables[position].sumOutFront();
      while (ft.windowSize() > prevLabels.length + 1) {
        ft = ft.sumOutFront();
      }
      return ft.conditionalLogProbGivenPrevious(prevLabels, label);
    } else {
      int[] p = new int[windowSize-1];
      System.arraycopy(prevLabels, prevLabels.length - p.length, p, 0, p.length);
      return factorTables[position].conditionalLogProbGivenPrevious(p, label);
    }
  }

  public double condLogProbGivenPrevious(int position, Object label, Object[] prevLabels) {
    return condLogProbGivenPrevious(position, classIndex.indexOf(label), objectArrayToIntArray(prevLabels));
  }

  public double condProbGivenPrevious(int position, int label, int[] prevLabels) {
    return Math.exp(condLogProbGivenPrevious(position, label, prevLabels));
  }

  public double condProbGivenPrevious(int position, Object label, Object[] prevLabels) {
    return Math.exp(condLogProbGivenPrevious(position, label, prevLabels));
  }

  public ClassicCounter condLogProbsGivenPrevious(int position, int[] prevlabels) {
    ClassicCounter c = new ClassicCounter();
    for (int i = 0; i < classIndex.size(); i++) {
      Object label = classIndex.get(i);
      c.incrementCount(label, condLogProbGivenPrevious(position, i, prevlabels));
    }
    return c;
  }

  public ClassicCounter condLogProbsGivenPrevious(int position, Object[] prevlabels) {
    ClassicCounter c = new ClassicCounter();
    for (int i = 0; i < classIndex.size(); i++) {
      Object label = classIndex.get(i);
      c.incrementCount(label, condLogProbGivenPrevious(position, i, prevlabels));
    }
    return c;
  }

  //
  // PROB OF TAG AT SINGLE POSITION CONDITIONED ON FOLLOWING SEQUENCE OF LABELS
  //

 public double condLogProbGivenNext(int position, int label, int[] nextLabels) {
   position = position+nextLabels.length;
    if (nextLabels.length + 1 == windowSize) {
      return factorTables[position].conditionalLogProbGivenNext(nextLabels, label);
    } else if (nextLabels.length + 1 < windowSize) {
      FactorTable ft = factorTables[position].sumOutFront();
      while (ft.windowSize() > nextLabels.length + 1) {
        ft = ft.sumOutFront();
      }
      return ft.conditionalLogProbGivenPrevious(nextLabels, label);
    } else {
      int[] p = new int[windowSize-1];
      System.arraycopy(nextLabels, 0, p, 0, p.length);
      return factorTables[position].conditionalLogProbGivenPrevious(p, label);
    }
  }
 
  public double condLogProbGivenNext(int position, Object label, Object[] nextLabels) {
    return condLogProbGivenNext(position, classIndex.indexOf(label), objectArrayToIntArray(nextLabels));
  }

  public double condProbGivenNext(int position, int label, int[] nextLabels) {
    return Math.exp(condLogProbGivenNext(position, label, nextLabels));
  }

  public double condProbGivenNext(int position, Object label, Object[] nextLabels) {
    return Math.exp(condLogProbGivenNext(position, label, nextLabels));
  }

  public ClassicCounter condLogProbsGivenNext(int position, int[] nextlabels) {
    ClassicCounter c = new ClassicCounter();
    for (int i = 0; i < classIndex.size(); i++) {
      Object label = classIndex.get(i);
      c.incrementCount(label, condLogProbGivenNext(position, i, nextlabels));
    }
    return c;
  }

  public ClassicCounter condLogProbsGivenNext(int position, Object[] nextlabels) {
    ClassicCounter c = new ClassicCounter();
    for (int i = 0; i < classIndex.size(); i++) {
      Object label = classIndex.get(i);
      c.incrementCount(label, condLogProbGivenNext(position, i, nextlabels));
    }
    return c;
  }

  //
  // PROB OF TAG AT SINGLE POSITION CONDITIONED ON PREVIOUS AND FOLLOWING SEQUENCE OF LABELS
  //

//   public double condProbGivenPreviousAndNext(int position, int label, int[] prevLabels, int[] nextLabels) {
    
//   }


  //
  // JOINT CONDITIONAL PROBS
  //

  /**
   * @param weights
   * @param data
   * @param labelIndices
   * @param numClasses
   * @param classIndex
   * @return a new CRFCliqueTree for the weights on the data
   */
  public static CRFCliqueTree getCalibratedCliqueTree(double[][] weights, int[][][] data, Index[] labelIndices, int numClasses, Index classIndex, String backgroundSymbol) {


    FactorTable[] factorTables = new FactorTable[data.length];
    FactorTable[] messages = new FactorTable[data.length - 1];

    for (int i = 0; i < data.length; i++) {

      factorTables[i] = getFactorTable(weights, data[i], labelIndices, numClasses);

      if (i > 0) {
        messages[i - 1] = factorTables[i - 1].sumOutFront();
        factorTables[i].multiplyInFront(messages[i - 1]);
      }
    }

    for (int i = factorTables.length - 2; i >= 0; i--) {

      FactorTable summedOut = factorTables[i + 1].sumOutEnd();
      summedOut.divideBy(messages[i]);
      factorTables[i].multiplyInEnd(summedOut);
    }

    return new CRFCliqueTree(factorTables, classIndex, backgroundSymbol);
  }

  private static FactorTable getFactorTable(double[][] weights, int[][] data, Index[] labelIndices, int numClasses) {

    FactorTable factorTable = null;

    for (int j = 0; j < labelIndices.length; j++) {
      Index labelIndex = labelIndices[j];
      FactorTable ft = new FactorTable(numClasses, j + 1);
            
      // ... and each possible labeling for that clique
      for (int k = 0, liSize = labelIndex.size(); k < liSize; k++) {
        int[] label = ((CRFLabel) labelIndex.get(k)).getLabel();
        double weight = 0.0;
        for (int m = 0; m < data[j].length; m++) {
          weight += weights[data[j][m]][k];
        }
        //      try{
        ft.setValue(label, weight);
        // } catch (Exception e) {
//          System.out.println("CRFCliqueTree::getFactorTable");
//          System.out.println("NumClasses: " + numClasses + " j+1: " + (j+1));
//          System.out.println("k: " + k+" label: "  +label+" labelIndexSize: " + labelIndex.size());
//          throw new RunTimeException(e.toString());
//      }

      }
      if (j > 0) {
        ft.multiplyInEnd(factorTable);
      }
      factorTable = ft;

    }

    return factorTable;

  }

  // SEQUENCE MODEL METHODS

  /**
   * Computes the distribution over values of the element at position pos in the sequence,
   * conditioned on the values of the elements in all other positions of the provided sequence.
   *
   * @param sequence the sequence containing the rest of the values to condition on
   * @param position      the position of the element to give a distribution for
   * @return an array of type double, representing a probability distribution; sums to 1.0
   */
  public double[] getConditionalDistribution(int[] sequence, int position) {
    double[] result = scoresOf(sequence, position);
    ArrayMath.logNormalize(result);
    // System.out.println("marginal:          " + ArrayMath.toString(marginal, nf));
    // System.out.println("conditional:       " + ArrayMath.toString(result, nf));
    result = ArrayMath.exp(result);
    // System.out.println("conditional:       " + ArrayMath.toString(result, nf));
    return result;
  }

  /**
   * Informs this sequence model that the value of the element at position pos has changed.
   * This allows this sequence model to update its internal model if desired.
   *
   * @param sequence
   * @param pos
   * @param oldVal
   */
  public void updateSequenceElement(int[] sequence, int pos, int oldVal) {
    // do nothing; we don't change this model
  }

  /**
   * Informs this sequence model that the value of the whole sequence is initialized to sequence
   *
   * @param sequence
   */
  public void setInitialSequence(int[] sequence) {
    // do nothing
  }

  /**
   * @return the number of possible values for each element; it is assumed
   *         to be the same for the element at each position
   */
  public int getNumValues() {
    return numClasses;
  }

}
