package edu.stanford.nlp.ie.crf;

import edu.stanford.nlp.ling.Datum;

import java.io.Serializable;
import java.util.List;


/**
 * The representation of Datums used internally in CRFClassifier.
 * 
 * @author Jenny Finkel
 */

public class CRFDatum<FEAT extends Serializable,LAB extends Serializable> implements Serializable {

  /**
   * Features for this Datum.
   */
  private final List<FEAT> features;
  private LAB label; // = null;

  /**
   * Constructs a new BasicDatum with the given features and label.
   *
   * @param features The features of the CRFDatum
   * @param label The label of the CRFDatum
   */
  public CRFDatum(List<FEAT> features, LAB label) {
    this(features);
    setLabel(label);
  }

  /**
   * Constructs a new BasicDatum with the given features and no labels.
   * @param features The features of the CRFDatum
   */
  public CRFDatum(List<FEAT> features) {
    this.features = features;
  }

  /**
   * Returns the collection that this BasicDatum was constructed with.
   *
   * @return the collection that this BasicDatum was constructed with.
   */
  public List<FEAT> asFeatures() {
    return features;
  }

  /**
   * Returns the label for this Datum, or null if none have been set.
   * @return The label for this Datum, or null if none have been set.
   */

  public LAB label() {
    return label;
  }

  /**
   * Removes all currently assigned Labels for this Datum then adds the
   * given Label.
   * Calling <tt>setLabel(null)</tt> effectively clears all labels.
   * @param label New label for CRFDatum
   */
  public void setLabel(LAB label) {
    this.label = label;
  }

  /**
   * Returns a String representation of this BasicDatum (lists features and labels).
   */
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("CRFDatum[\n");
    sb.append("     label=").append(label).append("\n");
    for (int i = 0, sz = features.size(); i < sz; i++) {
      sb.append("     features(").append(i).append("):").append(features.get(i)).append("\n");
    }
    sb.append("]");
    return sb.toString();
  }


  /**
   * Returns whether the given Datum contains the same features as this Datum.
   * Doesn't check the labels, should we change this?
   *
   * @param o The object to test equality with
   * @return Whether it is equal to this CRFDatum in terms of features
   */
  @Override
  public boolean equals(Object o) {
    if (!(o instanceof Datum)) {
      return (false);
    }

    Datum d = (Datum) o;
    return features.equals(d.asFeatures());
  }

  @Override
  public int hashCode() {
    return features.hashCode();
  }

  private static final long serialVersionUID = -8345554365027671190L;

}

