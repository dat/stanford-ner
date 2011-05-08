package edu.stanford.nlp.sequences;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.fsm.DFSA;

import java.util.List;
import java.io.PrintWriter;

/**
 * This interface is used for writing
 * lattices out of {@link SequenceClassifier}s.
 * 
 * @author Michel Galley
 */

public interface LatticeWriter {
  
  /**
   * This method prints the output lattice (typically, Viterbi search graph) of 
   * the classifier to a {@link PrintWriter}.
   */
  public void printLattice(DFSA tagLattice, List<CoreLabel> doc, PrintWriter out) ;
  
}
