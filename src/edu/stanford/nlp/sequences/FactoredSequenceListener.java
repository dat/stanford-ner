package edu.stanford.nlp.sequences;

/**
 * @author grenager
 *         Date: Apr 18, 2005
 */
public class FactoredSequenceListener implements SequenceListener {

  SequenceListener model1;
  SequenceListener model2;

  /**
   * Informs this sequence model that the value of the element at position pos has changed.
   * This allows this sequence model to update its internal model if desired.
   *
   * @param sequence
   * @param pos
   * @param oldVal
   */
  public void updateSequenceElement(int[] sequence, int pos, int oldVal) {
    model1.updateSequenceElement(sequence, pos, 0);
    model2.updateSequenceElement(sequence, pos, 0);
  }

  /**
   * Informs this sequence model that the value of the whole sequence is initialized to sequence
   *
   * @param sequence
   */
  public void setInitialSequence(int[] sequence) {
    model1.setInitialSequence(sequence);
    model2.setInitialSequence(sequence);
  }

  public FactoredSequenceListener(SequenceListener model1, SequenceListener model2) {
    this.model1 = model1;
    this.model2 = model2;
  }
}
