package edu.stanford.nlp.util;

public class IntPair extends IntTuple {


  public IntPair() {
    elements = new int[2];
  }


  public IntPair(int src, int trgt) {
    elements = new int[2];
    elements[0] = src;
    elements[1] = trgt;

  }


  public int getSource() {
    return get(0);
  }

  public int getTarget() {
    return get(1);
  }


  @Override
  public int hashCode() {
    return (65536 * elements[0]) ^ elements[1];
  }


  @Override
  public IntTuple getCopy() {
    IntPair nT = new IntPair(elements[0], elements[1]);
    return nT;
  }


}


