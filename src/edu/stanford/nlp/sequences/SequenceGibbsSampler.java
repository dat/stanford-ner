package edu.stanford.nlp.sequences;

import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.ling.HasWord;

import java.util.Random;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.PrintStream;

// TODO: change so that it uses the scoresOf() method properly

/**
 * A Gibbs sampler for sequence models. Given a sequence model implementing the SequenceModel
 * interface, this class is capable of
 * sampling sequences from the distribution over sequences that it defines. It can also use
 * this sampling procedure to find the best sequence.
 * @author grenager
 */
public class SequenceGibbsSampler implements BestSequenceFinder {

  // a random number generator
  private static Random random = new Random();
  public static int verbose = 0;

  private List document;
  private int numSamples;
  private int sampleInterval;
  private SequenceListener listener;

  public boolean returnLastFoundSequence = false;

  public static int[] copy(int[] a) {
    int[] result = new int[a.length];
    System.arraycopy(a, 0, result, 0, a.length);
    return result;
  }

  public static int[] getRandomSequence(SequenceModel model) {
    int[] result = new int[model.length()];
    for (int i = 0; i < result.length; i++) {
      int[] classes = model.getPossibleValues(i);
      result[i] = classes[random.nextInt(classes.length)];
    }
    return result;
  }

  /**
   * Finds the best sequence by collecting numSamples samples, scoring them, and then choosing
   * the highest scoring sample.
   * @return the array of type int representing the highest scoring sequence
   */
  public int[] bestSequence(SequenceModel model) {
    int[] initialSequence = getRandomSequence(model);
    return findBestUsingSampling(model, numSamples, sampleInterval, initialSequence);
  }

  /**
   * Finds the best sequence by collecting numSamples samples, scoring them, and then choosing
   * the highest scoring sample.
   * @param numSamples
   * @param sampleInterval
   * @return the array of type int representing the highest scoring sequence
   */
  public int[] findBestUsingSampling(SequenceModel model, int numSamples, int sampleInterval, int[] initialSequence) {
    List samples = collectSamples(model, numSamples, sampleInterval, initialSequence);
    int[] best = null;
    double bestScore = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < samples.size(); i++) {
      int[] sequence = (int[]) samples.get(i);
      double score = model.scoreOf(sequence);
      if (score>bestScore) {
        best = sequence;
        bestScore = score;
        System.err.println("found new best ("+bestScore+")");
        System.err.println(ArrayMath.toString(best));
      }
    }
    return best;
  }

  public int[] findBestUsingAnnealing(SequenceModel model, CoolingSchedule schedule) {
    int[] initialSequence = getRandomSequence(model);
    return findBestUsingAnnealing(model, schedule, initialSequence);
  }

  public int[] findBestUsingAnnealing(SequenceModel model, CoolingSchedule schedule, int[] initialSequence) {
    if (verbose>0) System.err.println("Doing annealing");
    listener.setInitialSequence(initialSequence);
    List result = new ArrayList();
    int[] sequence = initialSequence;
    int[] best = null;
    double bestScore = Double.NEGATIVE_INFINITY;
    double score = Double.NEGATIVE_INFINITY;
    if (!returnLastFoundSequence) {
      score = model.scoreOf(sequence);
    }

    for (int i=0; i<schedule.numIterations(); i++) {
      sequence = copy(sequence); // so we don't change the initial, or the one we just stored
      double temperature = schedule.getTemperature(i);
      sampleSequenceForward(model, sequence, temperature); // modifies tagSequence
      result.add(sequence);
      if (returnLastFoundSequence) {
        best = sequence;
      } else {
        score = model.scoreOf(sequence);
        //System.err.println(i+" "+score+" "+Arrays.toString(sequence));
        if (score>bestScore) {
          best = sequence;
          bestScore = score;
        }      
      }
      if (verbose>0) System.err.print(".");
    }
    if (verbose>1) {
      System.err.println();
      printSamples(result, System.err);
    }
    if (verbose>0) System.err.println("done.");
    //return sequence;
    return best;
  }

  /**
   * Collects numSamples samples of sequences, from the distribution over sequences defined
   * by the sequence model passed on construction.
   * All samples collected are sampleInterval samples apart, in an attempt to reduce
   * autocorrelation.
   * @param numSamples
   * @param sampleInterval
   * @return a List containing the sequence samples, as arrays of type int, and their scores
   */
  public List<int[]> collectSamples(SequenceModel model, int numSamples, int sampleInterval) {
    int[] initialSequence = getRandomSequence(model);
    return collectSamples(model, numSamples, sampleInterval, initialSequence);
  }

  /**
   * Collects numSamples samples of sequences, from the distribution over sequences defined
   * by the sequence model passed on construction.
   * All samples collected are sampleInterval samples apart, in an attempt to reduce
   * autocorrelation.
   * @param numSamples
   * @param sampleInterval
   * @param initialSequence
   * @return a Counter containing the sequence samples, as arrays of type int, and their scores
   */
  public List<int[]> collectSamples(SequenceModel model, int numSamples, int sampleInterval, int[] initialSequence) {
    if (verbose>0) System.err.print("Collecting samples");
    listener.setInitialSequence(initialSequence);
    List<int[]> result = new ArrayList<int[]>();
    int[] sequence = initialSequence;
    for (int i=0; i<numSamples; i++) {
      sequence = copy(sequence); // so we don't change the initial, or the one we just stored
      sampleSequenceRepeatedly(model, sequence, sampleInterval); // modifies tagSequence
      result.add(sequence); // save it to return later
      if (verbose>0) System.err.print(".");
      System.err.flush();
    }
    if (verbose>1) {
      System.err.println();
      printSamples(result, System.err);
    }
    if (verbose>0) System.err.println("done.");
    return result;
  }

  /**
   * Samples the sequence repeatedly, making numSamples passes over the entire sequence.
   * @param sequence
   * @param numSamples
   */
  public void sampleSequenceRepeatedly(SequenceModel model, int[] sequence, int numSamples) {
    sequence = copy(sequence); // so we don't change the initial, or the one we just stored
    listener.setInitialSequence(sequence);
    for (int iter=0; iter<numSamples; iter++) {
      sampleSequenceForward(model, sequence);
    }
  }

  /**
   * Samples the sequence repeatedly, making numSamples passes over the entire sequence.
   * Destructively modifies the sequence in place.
   * @param numSamples
   */
  public void sampleSequenceRepeatedly(SequenceModel model, int numSamples) {
    int[] sequence = getRandomSequence(model);
    sampleSequenceRepeatedly(model, sequence, numSamples);
  }

  /**
   * Samples the complete sequence once in the forward direction
   * Destructively modifies the sequence in place.
   * @param sequence the sequence to start with.
   */
  public void sampleSequenceForward(SequenceModel model, int[] sequence) {
    sampleSequenceForward(model, sequence, 1.0);
  }
  /**
   * Samples the complete sequence once in the forward direction
   * Destructively modifies the sequence in place.
   * @param sequence the sequence to start with.
   */
  public void sampleSequenceForward(SequenceModel model, int[] sequence, double temperature) {
    // System.err.println("Sampling forward");
    for (int pos=0; pos<sequence.length; pos++) {
      samplePosition(model, sequence, pos, temperature);
    }
  }

  /**
   * Samples the complete sequence once in the backward direction
   * Destructively modifies the sequence in place.
   * @param sequence the sequence to start with.
   */
  public void sampleSequenceBackward(SequenceModel model, int[] sequence) {
    sampleSequenceBackward(model, sequence, 1.0);
  }
  /**
   * Samples the complete sequence once in the backward direction
   * Destructively modifies the sequence in place.
   * @param sequence the sequence to start with.
   */
  public void sampleSequenceBackward(SequenceModel model, int[] sequence, double temperature) {
    for (int pos=sequence.length-1; pos>=0; pos--) {
      samplePosition(model, sequence, pos, temperature);
    }
  }

  /**
   * Samples a single position in the sequence.
   * Destructively modifies the sequence in place.
   * returns the score of the new sequence
   * @param sequence the sequence to start with
   * @param pos the position to sample.
   */
  public double samplePosition(SequenceModel model, int[] sequence, int pos) {
    return samplePosition(model, sequence, pos, 1.0);
  }

  /**
   * Samples a single position in the sequence.
   * Destructively modifies the sequence in place.
   * returns the score of the new sequence
   * @param sequence the sequence to start with
   * @param pos the position to sample.
   */
  public double samplePosition(SequenceModel model, int[] sequence, int pos, double temperature) {
    double[] distribution = model.scoresOf(sequence, pos);
    if (temperature!=1.0) {
      if (temperature==0.0) {
        // set the max to 1.0
        int argmax = ArrayMath.argmax(distribution);
        Arrays.fill(distribution, Double.NEGATIVE_INFINITY);
        distribution[argmax] = 0.0;
      } else {
        // take all to a power
        // use the temperature to increase/decrease the entropy of the sampling distribution
        ArrayMath.multiplyInPlace(distribution, 1.0/temperature);
      }
    }
    ArrayMath.logNormalize(distribution);
    ArrayMath.expInPlace(distribution);
    int oldTag = sequence[pos];
    int newTag = ArrayMath.sampleFromDistribution(distribution, random);
//    System.out.println("Sampled " + oldTag + "->" + newTag);
    sequence[pos] = newTag;
    listener.updateSequenceElement(sequence, pos, oldTag);
    return distribution[newTag];
  }

  public void printSamples(List samples, PrintStream out) {
    for (int i = 0; i < document.size(); i++) {
      HasWord word = (HasWord) document.get(i);
      String s = "null";
      if (word!=null) {
        s = word.word();
      }
      out.print(StringUtils.padOrTrim(s, 10));
      for (int j = 0; j < samples.size(); j++) {
        int[] sequence = (int[]) samples.get(j);
        out.print(" " + StringUtils.padLeft(sequence[i], 2));
      }
      out.println();
    }
  }

  /**
   * @param numSamples
   * @param sampleInterval
   * @param document the underlying document which is a list of HasWord; a slight abstraction violation, but useful for debugging!!
   */
  public SequenceGibbsSampler(int numSamples, int sampleInterval, SequenceListener listener, List document, boolean returnLastFoundSequence) {
    this.numSamples = numSamples;
    this.sampleInterval = sampleInterval;
    this.listener = listener;
    this.document = document;
    this.returnLastFoundSequence = returnLastFoundSequence;
  }

  public SequenceGibbsSampler(int numSamples, int sampleInterval, SequenceListener listener, List document) {
    this(numSamples, sampleInterval, listener, document, false);
  }

  public SequenceGibbsSampler(int numSamples, int sampleInterval, SequenceListener listener) {
    this(numSamples, sampleInterval, listener, null);
  }
}
