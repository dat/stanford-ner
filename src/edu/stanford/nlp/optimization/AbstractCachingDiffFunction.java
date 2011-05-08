package edu.stanford.nlp.optimization;

import java.util.Arrays;

/**
 * @author Dan Klein
 */

public abstract class AbstractCachingDiffFunction implements DiffFunction, HasInitial {

  double[] lastX = null;
  int fevals = 0;
  protected double[] derivative = null;
  protected double value = 0.0;


  abstract public int domainDimension();

  /**
   * Calculate the value at x and the derivative and save them in the respective fields
   *
   * @param x
   */
  abstract protected void calculate(double[] x);

  /**
   * Clears the cache in a way that doesn't require reallocation :-)
   */
  protected void clearCache() {
    if (lastX != null) lastX[0] = Double.NaN;
  }

  public double[] initial() {
    double[] initial = new double[domainDimension()];
    Arrays.fill(initial, 0.0);
    return initial;
  }

  protected void copy(double[] y, double[] x) {
    System.arraycopy(x, 0, y, 0, x.length);
  }

  void ensure(double[] x) {
    if (Arrays.equals(x, lastX)) {
      return;
    }
    if (lastX == null) {
      lastX = new double[domainDimension()];
    }
    if (derivative == null) {
      derivative = new double[domainDimension()];
    }
    copy(lastX, x);
    fevals += 1;
    calculate(x);
  }

  public double valueAt(double[] x) {
    ensure(x);
    return value;
  }

  public double[] derivativeAt(double[] x) {
    ensure(x);
    return derivative;
  }

  public double lastValue() {
    return value;
  }

	public void setValue(double v) {
			value = v;
	}
}
