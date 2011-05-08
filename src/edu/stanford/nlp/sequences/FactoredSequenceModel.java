package edu.stanford.nlp.sequences;

import edu.stanford.nlp.math.ArrayMath;

/**
 * @author grenager
 *         Date: Dec 14, 2004
 */
public class FactoredSequenceModel implements SequenceModel {

  SequenceModel model1;
  SequenceModel model2;

  /**
   * Computes the distribution over values of the element at position pos in the sequence,
   * conditioned on the values of the elements in all other positions of the provided sequence.
   *
   * @param sequence the sequence containing the rest of the values to condition on
   * @param pos      the position of the element to give a distribution for
   * @return an array of type double, representing a probability distribution; must sum to 1.0
   */
  public double[] scoresOf(int[] sequence, int pos) {
    double[] dist1 = model1.scoresOf(sequence, pos);
    double[] dist2 = model2.scoresOf(sequence, pos);
    double[] dist = ArrayMath.pairwiseAdd(dist1, dist2);

//     if (pos > 0 && pos < sequence.length - 1) {
//       System.err.println("position: "+pos+" ["+sequence[pos-1]+","+sequence[pos]+","+sequence[pos+1]+"]");
//       System.err.println(java.util.Arrays.toString(sequence));
//       for (int i = 0; i < dist.length; i++) {
//         System.err.println(i+": "+dist1[i]+" "+dist2[i]+" "+dist[i]);
//       }
//       System.err.println();
//     }

    return dist;
  }

  public double scoreOf(int[] sequence, int pos) {
    return scoresOf(sequence, pos)[sequence[pos]];
  }

  /**
   * Computes the score assigned by this model to the provided sequence. Typically this will be a
   * probability in log space (since the probabilities are small).
   *
   * @param sequence the sequence to compute a score for
   * @return the score for the sequence
   */
  public double scoreOf(int[] sequence) {
    return model1.scoreOf(sequence) + model2.scoreOf(sequence);
  }

  /**
   * @return the length of the sequence
   */
  public int length() {
    return model1.length();
  }

  public int leftWindow() {
    return model1.leftWindow();
  }

  public int rightWindow() {
    return 0;  //To change body of implemented methods use File | Settings | File Templates.
  }

  public int[] getPossibleValues(int position) {
    return model1.getPossibleValues(position);
  }

  public FactoredSequenceModel(SequenceModel model1, SequenceModel model2) {
    //if (model1.leftWindow() != model2.leftWindow()) throw new RuntimeException("Two models must have same window size");
    if (model1.getPossibleValues(0).length != model2.getPossibleValues(0).length) throw new RuntimeException("Two models must have the same number of classes");
    if (model1.length() != model2.length()) throw new RuntimeException("Two models must have the same sequence length");
    this.model1 = model1;
    this.model2 = model2;
  }
}
