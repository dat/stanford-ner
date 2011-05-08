package edu.stanford.nlp.util;

import edu.stanford.nlp.ling.HasWord;

import java.io.Serializable;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * Represents a String with a corresponding integer ID.
 * Keeps a static index of all the Strings, indexed by ID.
 * 
 * @author danielcer
 *
 */
public class IString implements CharSequence, Serializable, HasIntegerIdentity, HasWord {
  // TODOX make serialization clean
  public static final IndexInterface<String> index = new OAIndex<String>();

  private final String stringRep;
  public final int id;

  private enum Classing { NONE, BACKSLASH, IBM }
  private static final Classing classing = Classing.IBM;

  public IString() {
    id = -1;
    stringRep = "";
  }
  /**
   *
   * @param string
   */
  public IString(String string) {
    if(classing == Classing.BACKSLASH) { // e.g., on december 4\\num
      int doubleBackSlashPos = string.indexOf("\\\\");
      if (doubleBackSlashPos != -1) {
        stringRep = string.substring(0, doubleBackSlashPos);
        id = index.indexOf(string.substring(doubleBackSlashPos), true);
        return;
      }
    } else if(classing == Classing.IBM) { // e.g., on december $num_(4)
      if(string.length() > 2 && string.startsWith("$")) {
        int delim = string.indexOf("_(");
        if(delim != -1 && string.endsWith(")")) {
          stringRep = string.substring(delim+2,string.length()-1);
          id = index.indexOf(string.substring(0,delim), true);
        } else {
          stringRep = string;
          id = index.indexOf(string, true);
        }
        return;
      }
    }
    stringRep = string;
    id = index.indexOf(stringRep, true);
  }

  /**
   *
   * @param id
   */
  public IString(int id) {
    this.id = id;
    stringRep = index.get(id);
  }

  /**
   *
   */
  private static final long serialVersionUID = 2718L;

  public char charAt(int index) {
    return stringRep.charAt(index);
  }

  public int length() {
    return stringRep.length();
  }


  public CharSequence subSequence(int start, int end) {
    return stringRep.subSequence(start, end);
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof IString)) {
      System.err.printf("o class: %s\n", o.getClass());
      throw new UnsupportedOperationException();
    }
    IString istr = (IString)o;
    return this.id == istr.id;
  }

  public long longHashCode() {
    return id;
  }

  @Override
  public int hashCode() {
    return id;
  }

  @Override
  public String toString() {
    return stringRep;
  }

  public int getId() {
    return id;
  }

  public static String getString(int id) {
    return index.get(id);
  }

  public String word() {
    return toString();
  }

  public void setWord(String word) {
    throw new UnsupportedOperationException();
  }
  
  static private WrapperIndex wrapperIndex; // = null;

  static public IndexInterface<IString> identityIndex() {
    if (wrapperIndex == null) {
      wrapperIndex = new WrapperIndex();
    }
    return wrapperIndex;
  }

  static private class WrapperIndex implements IndexInterface<IString> {

    /**
     *
     */
    private static final long serialVersionUID = 2718L;

    public boolean contains(Object o) {
      if (!(o instanceof IString)) return false;
      IString istring = (IString)o;
      return index.contains(istring.stringRep);
    }

    public IString get(int i) {
      return new IString(index.get(i));
    }

    public int indexOf(IString o) {
      return index.indexOf(o.stringRep);
    }

    public int indexOf(IString o, boolean add) {
      return index.indexOf(o.stringRep, add);
    }

    public int size() {
      return index.size();
    }

  }

}

