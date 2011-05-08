package edu.stanford.nlp.util;

import java.util.*;

/**
 * Implements a finite beam, taking a comparator (default is
 * ScoredComparator.ASCENDING_COMPARATOR, the MAX object according to
 * the comparator is the one to be removed) and a beam size on
 * construction (default is 100).  Adding an object may cause the
 * worst-scored object to be removed from the beam (and that object
 * may well be the newly added object itself).
 *
 * @author <a href="mailto:klein@cs.stanford.edu">Dan Klein</a>
 * @version 1.0
 */
public class Beam<T> extends AbstractSet<T> {

  protected int maxBeamSize;
  protected Heap<T> elements;

  public int capacity() {
    return maxBeamSize;
  }

  @Override
  public int size() {
    return elements.size();
  }

  @Override
  public Iterator<T> iterator() {
    return asSortedList().iterator();
  }

  public List<T> asSortedList() {
    LinkedList<T> list = new LinkedList<T>();
    for (Iterator<T> i = elements.iterator(); i.hasNext();) {
      list.addFirst(i.next());
    }
    return list;
  }

  @Override
  public boolean add(T o) {
    boolean added = true;
    elements.add(o);
    while (size() > capacity()) {
      Object dumped = elements.extractMin();
      if (dumped.equals(o)) {
        added = false;
      }
    }
    return added;
  }

  @Override
  public boolean remove(Object o) {
    //return elements.remove(o);
    throw new UnsupportedOperationException();
  }

  public Beam() {
    this(100);
  }

  public Beam(int maxBeamSize) {
    this(maxBeamSize, ScoredComparator.ASCENDING_COMPARATOR);
  }

  public Beam(int maxBeamSize, Comparator cmp) {
    elements = new ArrayHeap<T>(cmp);
    this.maxBeamSize = maxBeamSize;
  }

  public static void main(String[] args) {
    Beam<ScoredObject> b = new Beam<ScoredObject>(2, ScoredComparator.ASCENDING_COMPARATOR);
    b.add(new ScoredObject("1", 1.0));
    b.add(new ScoredObject("2", 2.0));
    b.add(new ScoredObject("3", 3.0));
    b.add(new ScoredObject("0", 0.0));
    for (Iterator<ScoredObject> bI = b.iterator(); bI.hasNext();) {
      ScoredObject sO = bI.next();
      System.out.println(sO);
    }
  }

}
