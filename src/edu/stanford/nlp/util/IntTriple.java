package edu.stanford.nlp.util;

public class IntTriple extends IntTuple {


  public IntTriple() {
    elements = new int[3];

  }

  public IntTriple(int src, int mid, int trgt) {
    elements = new int[3];
    elements[0] = src;
    elements[1] = mid;
    elements[2] = trgt;
  }


  @Override
  public IntTuple getCopy() {
    IntTriple nT = new IntTriple(elements[0], elements[1], elements[2]);
    return nT;
  }


  public int getSource() {
    return elements[0];
  }

  public int getTarget() {
    return elements[2];
  }

  public int getMiddle() {
    return elements[1];
  }

  @Override
  public int hashCode() {
    return (elements[0] << 20) ^ (elements[1] << 10) ^ (elements[2]);
  }


}

