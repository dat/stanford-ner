package edu.stanford.nlp.ie.crf;

import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.math.SloppyMath;
import edu.stanford.nlp.util.Index;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/** Stores a factor table as a one dimensional array of doubles.
 *
 *  @author Jenny Finkel
 */
public class FactorTable {

  private int numClasses;
  private int windowSize;

  private double[] table;

  public FactorTable(int numClasses, int windowSize) {
    this.numClasses = numClasses;
    this.windowSize = windowSize;

    table = new double[SloppyMath.intPow(numClasses, windowSize)];
    Arrays.fill(table, Double.NEGATIVE_INFINITY);
  }

  public boolean containsNaN() {
    for (int i = 0; i < table.length; i++) {
      if (Double.isNaN(table[i])) {
        return true;
      }
    }
    return false;
  }


  public String toProbString() {
    StringBuilder sb = new StringBuilder("{\n");
    for (int i = 0; i < table.length; i++) {
      sb.append(Arrays.toString(toArray(i)));
      sb.append(": ");
      sb.append(prob(toArray(i)));
      sb.append("\n");
    }
    sb.append("}");
    return sb.toString();
  }

  public String toString(Index classIndex) {
    StringBuilder sb = new StringBuilder("{\n");
    for (int i = 0; i < table.length; i++) {
      sb.append(toString(toArray(i), classIndex));
      sb.append(": ");
      sb.append(getValue(i));
      sb.append("\n");
    }
    sb.append("}");
    return sb.toString();
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("{\n");
    for (int i = 0; i < table.length; i++) {
      sb.append(Arrays.toString(toArray(i)));
      sb.append(": ");
      sb.append(getValue(i));
      sb.append("\n");
    }
    sb.append("}");
    return sb.toString();
  }

  private static String toString(int[] array, Index classIndex) {
    List l = new ArrayList();
    for (int i = 0; i < array.length; i++) {
      l.add(classIndex.get(array[i]));
    }
    return l.toString();
  }

  private int[] toArray(int index) {
    int[] indices = new int[windowSize];
    for (int i = indices.length - 1; i >= 0; i--) {
      indices[i] = index % numClasses;
      index /= numClasses;
    }
    return indices;
  }

  private int indexOf(int[] entry) {
    int index = 0;
    for (int i = 0; i < entry.length; i++) {
      index *= numClasses;
      index += entry[i];
    }
    if (index<0) throw new RuntimeException("index=" + index + " entry=" + Arrays.toString(entry));
    return index;
  }

  private int indexOf(int[] front, int end) {
    int index = 0;
    for (int i = 0; i < front.length; i++) {
      index *= numClasses;
      index += front[i];
    }
    index *= numClasses;
    index += end;
    return index;
  }

  private int indexOf(int front, int[] end) {
    int index = front;
    for (int i = 0; i < end.length; i++) {
      index *= numClasses;
      index += end[i];
    }
    return index;
  }

  private int[] indicesEnd(int[] entries) {
    int[] indices = new int[SloppyMath.intPow(numClasses, windowSize - entries.length)];
    int offset = SloppyMath.intPow(numClasses, entries.length);
    int index = 0;
    for (int i = 0; i < entries.length; i++) {
      index *= numClasses;
      index += entries[i];
    }
    for (int i = 0; i < indices.length; i++) {
      indices[i] = index;
      index += offset;
    }
    return indices;
  }

  private int[] indicesFront(int[] entries) {
    int[] indices = new int[SloppyMath.intPow(numClasses, windowSize - entries.length)];
    int offset = SloppyMath.intPow(numClasses, windowSize - entries.length);
    int start = 0;
    for (int i = 0; i < entries.length; i++) {
      start *= numClasses;
      start += entries[i];
    }
    start *= offset;
    int end = 0;
    for (int i = 0; i < entries.length; i++) {
      end *= numClasses;
      end += entries[i];
      if (i == entries.length - 1) {
        end += 1;
      }
    }
    end *= offset;
    for (int i = start; i < end; i++) {
      indices[i - start] = i;
    }
    return indices;
  }

  public int windowSize() {
    return windowSize;
  }

  public int numClasses() {
    return numClasses;
  }

  private int size() {
    return table.length;
  }

  public double totalMass() {
    return ArrayMath.logSum(table);
  }

  public double unnormalizedLogProb(int[] label) {
    return getValue(label);
  }

  public double logProb(int[] label) {
    return unnormalizedLogProb(label) - totalMass();
  }


  public double prob(int[] label) {
    return Math.pow(Math.E, unnormalizedLogProb(label) - totalMass());
  }

  /**
   * Computes the probability of the tag OF being at the end of the table
   * given that the previous tag sequence in table is GIVEN.
   * given is at the begining, of is at the end
   * @param given
   * @param of
   * @return the probability of the tag OF being at the end of the table
   */
  public double conditionalLogProbGivenPrevious(int[] given, int of) {
    if (given.length != windowSize - 1) {
      System.err.println("error computing conditional log prob");
      System.exit(0);
    }
    int[] label = indicesFront(given);
    double[] masses = new double[label.length];
    for (int i = 0; i < masses.length; i++) {
      masses[i] = table[label[i]];
    }
    double z = ArrayMath.logSum(masses);

    int i = indexOf(given, of);

//     if (SloppyMath.isDangerous(z) || SloppyMath.isDangerous(table[i])) {
//       System.err.println("z="+z);
//       System.err.println("t="+table[i]);
//     }

    return table[i] - z;
  }

  /**
   * Computes the probabilities of the tag at the end of the table
   * given that the previous tag sequence in table is GIVEN.
   * given is at the begining, position in question is at the end
   * @param given
   * @return the probabilities of the tag at the end of the table
   */
  public double[] conditionalLogProbsGivenPrevious(int[] given) {
    if (given.length != windowSize - 1) {
      System.err.println("error computing conditional log prob");
      System.exit(0);
    }
    double[] result = new double[numClasses];
    for (int i = 0; i < numClasses; i++) {
      int index = indexOf(given, i);
      result[i] = table[index];
    }
    ArrayMath.logNormalize(result);
    return result;
  }

  /**
   * Computes the probability of the sequence OF being at the end of the table
   * given that the first tag in table is GIVEN.
   * given is at the begining, of is at the end
   * @param given
   * @param of
   * @return the probability of the sequence of being at the end of the table
   */
  public double conditionalLogProbGivenFirst(int given, int[] of) {
    if (of.length != windowSize - 1) {
      System.err.println("error computing conditional log prob");
      System.exit(0);
    }
    // compute P(given, of)
    int[] labels = new int[windowSize];
    labels[0] = given;
    System.arraycopy(of, 0, labels, 1, windowSize-1);
    //double probAll = logProb(labels);
    double probAll = unnormalizedLogProb(labels);

    // compute P(given)
    //double probGiven = logProbFront(given);
    double probGiven = unnormalizedLogProbFront(given);

    // compute P(given, of) / P(given)
    return probAll - probGiven;
  }

  /**
   * Computes the probability of the sequence OF being at the end of the table
   * given that the first tag in table is GIVEN.
   * given is at the begining, of is at the end
   * @param given
   * @param of
   * @return the probability of the sequence of being at the end of the table
   */
  public double unnormalizedConditionalLogProbGivenFirst(int given, int[] of) {
    if (of.length != windowSize - 1) {
      System.err.println("error computing conditional log prob");
      System.exit(0);
    }
    // compute P(given, of)
    int[] labels = new int[windowSize];
    labels[0] = given;
    System.arraycopy(of, 0, labels, 1, windowSize-1);
    //double probAll = logProb(labels);
    double probAll = unnormalizedLogProb(labels);

    // compute P(given)
    //double probGiven = logProbFront(given);
    //double probGiven = unnormalizedLogProbFront(given);

    // compute P(given, of) / P(given)
    //return probAll - probGiven;
    return probAll;
  }

  /**
   * Computes the probability of the tag OF being at the beginning of the table
   * given that the tag sequence GIVEN is at the end of the table.
   * given is at the end, of is at the begining
   * @param given
   * @param of
   * @return the probability of the tag of being at the beginning of the table
   */
  public double conditionalLogProbGivenNext(int[] given, int of) {
    if (given.length != windowSize - 1) {
      System.err.println("error computing conditional log prob");
      System.exit(0);
    }
    int[] label = indicesEnd(given);
    double[] masses = new double[label.length];
    for (int i = 0; i < masses.length; i++) {
      masses[i] = table[label[i]];
    }
    double z = ArrayMath.logSum(masses);

    return table[indexOf(of, given)] - z;
  }

  public double unnormalizedLogProbFront(int[] labels) {
    labels = indicesFront(labels);
    double[] masses = new double[labels.length];
    for (int i = 0; i < masses.length; i++) {
      masses[i] = table[labels[i]];
    }
    return ArrayMath.logSum(masses);
  }

  public double logProbFront(int[] label) {
    return unnormalizedLogProbFront(label) - totalMass();
  }

  public double unnormalizedLogProbFront(int label) {
    int[] labels = {label};
    return unnormalizedLogProbFront(labels);
  }

  public double logProbFront(int label) {
    return unnormalizedLogProbFront(label) - totalMass();
  }

  public double unnormalizedLogProbEnd(int[] labels) {
    labels = indicesEnd(labels);
    double[] masses = new double[labels.length];
    for (int i = 0; i < masses.length; i++) {
      masses[i] = table[labels[i]];
    }
    return ArrayMath.logSum(masses);
  }

  public double logProbEnd(int[] labels) {
    return unnormalizedLogProbEnd(labels) - totalMass();
  }

  public double unnormalizedLogProbEnd(int label) {
    int[] labels = {label};
    return unnormalizedLogProbEnd(labels);
  }

  public double logProbEnd(int label) {
    return unnormalizedLogProbEnd(label) - totalMass();
  }

  private double getValue(int index) {
    return table[index];
  }

  public double getValue(int[] label) {
    return table[indexOf(label)];
  }

  private void setValue(int index, double value) {
    table[index] = value;
  }

  public void setValue(int[] label, double value) {
      //      try{    
      table[indexOf(label)] = value;
     //  } catch (Exception e) {
// 	  e.printStackTrace();
// 	  System.err.println("Table length: " + table.length + " indexOf(label): " + indexOf(label));
// 	  throw new ArrayIndexOutOfBoundsException(e.toString());
// 	  // System.exit(1);
//       }
  }

  public void incrementValue(int[] label, double value) {
    table[indexOf(label)] += value;
  }

  private void logIncrementValue(int index, double value) {
    table[index] = SloppyMath.logAdd(table[index], value);
  }

  public void logIncrementValue(int[] label, double value) {
    int index = indexOf(label);
    table[index] = SloppyMath.logAdd(table[index], value);
  }

  public void multiplyInFront(FactorTable other) {
    int divisor = SloppyMath.intPow(numClasses, windowSize - other.windowSize());
    for (int i = 0; i < table.length; i++) {
      table[i] += other.getValue(i / divisor);
    }
  }

  public void multiplyInEnd(FactorTable other) {
    int divisor = SloppyMath.intPow(numClasses, other.windowSize());
    for (int i = 0; i < table.length; i++) {
      table[i] += other.getValue(i % divisor);
    }
  }

  public FactorTable sumOutEnd() {
    FactorTable ft = new FactorTable(numClasses, windowSize - 1);
    for (int i = 0; i < table.length; i++) {
      ft.logIncrementValue(i / numClasses, table[i]);
    }
    return ft;
  }

  public FactorTable sumOutFront() {
    FactorTable ft = new FactorTable(numClasses, windowSize - 1);
    int mod = SloppyMath.intPow(numClasses, windowSize - 1);
    for (int i = 0; i < table.length; i++) {
      ft.logIncrementValue(i % mod, table[i]);
    }
    return ft;
  }

  public void divideBy(FactorTable other) {
    for (int i = 0; i < table.length; i++) {
      if (table[i] != Double.NEGATIVE_INFINITY || other.table[i] != Double.NEGATIVE_INFINITY) {
        table[i] -= other.table[i];
      }
    }
  }

  public static void main(String[] args) {
    FactorTable ft = new FactorTable(6, 3);

    /**
     for (int i = 0; i < 2; i++) {
     for (int j = 0; j < 2; j++) {
     for (int k = 0; k < 2; k++) {
     int[] a = new int[]{i, j, k};
     System.out.print(ft.toString(a)+": "+ft.indexOf(a));
     }
     }
     }
     for (int i = 0; i < 2; i++) {
     int[] b = new int[]{i};
     System.out.print(ft.toString(b)+": "+ft.toString(ft.indicesFront(b)));
     }
     for (int i = 0; i < 2; i++) {
     for (int j = 0; j < 2; j++) {
     int[] b = new int[]{i, j};
     System.out.print(ft.toString(b)+": "+ft.toString(ft.indicesFront(b)));
     }
     }
     for (int i = 0; i < 2; i++) {
     int[] b = new int[]{i};
     System.out.print(ft.toString(b)+": "+ft.toString(ft.indicesBack(b)));
     }	for (int i = 0; i < 2; i++) {
     for (int j = 0; j < 2; j++) {
     int[] b = new int[]{i, j};
     ft2.setValue(b, (i*2)+j);
     }
     }
     for (int i = 0; i < 2; i++) {
     for (int j = 0; j < 2; j++) {
     int[] b = new int[]{i, j};
     System.out.print(ft.toString(b)+": "+ft.toString(ft.indicesBack(b)));
     }
     }

     System.out.println("##########################################");

     **/

    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        for (int k = 0; k < 6; k++) {
          int[] b = new int[]{i, j, k};
          ft.setValue(b, (i * 4) + (j * 2) + k);
        }
      }
    }

    //System.out.println(ft);
    //System.out.println(ft.sumOutFront());

    FactorTable ft2 = new FactorTable(6, 2);
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        int[] b = new int[]{i, j};
        ft2.setValue(b, i * 6 + j);
      }
    }

    System.out.println(ft);
    //FactorTable ft3 = ft2.sumOutFront();
    //System.out.println(ft3);

    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        int[] b = new int[]{i, j};
        double t = 0;
        for (int k = 0; k < 6; k++) {
          t += Math.pow(Math.E, ft.conditionalLogProbGivenPrevious(b, k));
          System.err.println(k + "|" + i + "," + j + " : " + Math.pow(Math.E, ft.conditionalLogProbGivenPrevious(b, k)));
        }
        System.out.println(t);
      }
    }
  }
}
