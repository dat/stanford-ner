package edu.stanford.nlp.linalg;

import java.util.Iterator;
import java.util.Set;

/**
 * Interface for dense or sparse arrays which store numbers.
 *
 * @author Sepandar Kamvar (sdkamvar@stanford.edu)
 */

public abstract interface Array extends Cloneable {

  /**
   * Sets column index of column vector.  the default is -1;
   */

  public abstract void setColumnIndex(int ci);

  /**
   * Returns column index of column vector
   */

  public abstract int columnIndex();

  /**
   * Sets all elements = <code>value</code>
   * (in sparse matrices, sets all nonzero elements = value)
   */

  public void setAll(double value);

  /**
   * Gets double value of element at index i
   */
  public abstract double get(int index);


  /**
   * Sets value of element at index i to val
   */
  public abstract void set(int i, double val);


  /**
   * Returns number of elements in Array
   */

  public abstract int size();


  /**
   * Returns a set view of the Entries contained in this array.
   * Note: this differs from java.util.AbstractMap.entrySet() in that
   * it returns a Set of Entries, rather than a set of objects.
   */

  public abstract Set entrySet();

  /**
   * Returns true if the array contains index <code>key</code>
   */

  public abstract boolean containsKey(int key);

  /**
   * Returns an iterator over the entries in the matrix
   */

  public abstract Iterator iterator();

  /**
   * Returns true if <code>o==this</code>
   */

  public abstract boolean equals(Object o);

  /**
   * Returns String representation of Array
   */

  public abstract String toString();

  /**
   * Returns deep copy of Array
   */

  public abstract Object clone();

  /**
   * Returns an empty Array of the same dynamic type with given size
   */

  public abstract Array newInstance(int size);

  //note, none of these methods change the reciever.

  /**
   * Returns <code>this + addend</code> does not change receiver.
   */

  public abstract Array add(Array addend);

  /**
   * Returns <code>this * factor</code> does not change receiver.
   */

  public abstract Array scale(double factor);

  /**
   * Returns L2 norm of vector
   */

  public abstract double norm();

  /**
   * Returns Euclidean distance from <code>this</code> to <code>other</code>
   */

  public abstract double distance(Array other);

  /**
   * Returns dot product of <code>this</code> with <code>other</code>
   */

  public abstract double dot(Array other);

  /**
   * Returns componentwise multiply
   */

  public abstract Array multiply(Array multiplicand);

  /**
   * Returns componentwise divide: this/dividend
   */

  public abstract Array divide(Array dividend);

  /**
   * Scalar multiply
   */

  public abstract Array multiply(double multiplicand);

  /**
   * Scalar divide
   */

  public abstract Array divide(double dividend);


  /**
   * Returns vector with euclidian distance normalized to 1.  does not change reciever.
   */

  public abstract Array normalize();

  /**
   * Returns number of nonzero entries in array
   */

  public abstract int cardinality();

  /**
   * Transforms the array to log space.  this is kind of a hack, so try to fix it later, by defining things like componentwise array functions, etc.
   * takes the log of each component in array
   */

  public abstract Array toLogSpace();

}


