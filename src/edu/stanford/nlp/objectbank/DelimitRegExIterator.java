package edu.stanford.nlp.objectbank;

import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.AbstractIterator;

import java.io.BufferedReader;
import java.io.Reader;
import java.io.StringReader;
import java.io.Serializable;
import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * An Iterator that reads the contents of a buffer, delimited by the specified
 * delimiter, and then be subsequently processed by an Function to produce
 * Objects of type T.
 *
 * @author Jenny Finkel <A HREF="mailto:jrfinkel@stanford.edu>jrfinkel@stanford.edu</A>
 * @param <T> The type of the objects returned
 */
public class DelimitRegExIterator<T> extends AbstractIterator<T> {

  private Iterator<String> tokens;
  private Function<String,T> op;
  private T nextToken; // = null;

  //TODO: not sure if this is the best way to name things...
  public static DelimitRegExIterator<String> defaultDelimitRegExIterator(Reader in, String delimiter) {
    return new DelimitRegExIterator<String>(in, delimiter, new IdentityFunction<String>());
  }

  public DelimitRegExIterator(Reader r, String delimiter, Function<String,T> op) {
    this.op = op;
    BufferedReader in = new BufferedReader(r);
    try {
      String line;
      StringBuilder input = new StringBuilder();
      while ((line = in.readLine()) != null) {
        input.append(line).append("\n");
      }
      line = input.toString();
      Pattern p = Pattern.compile("^"+delimiter);
      Matcher m = p.matcher(line);
      line = m.replaceAll("");
      p = Pattern.compile(delimiter+"$");
      m = p.matcher(line);
      line = m.replaceAll("");
      line = line.trim();

      tokens = Arrays.asList(line.split(delimiter)).iterator();
    } catch (Exception e) {
    }
    setNext();
  }

  private void setNext() {
    if (tokens.hasNext()) {
      String s = tokens.next();
      nextToken = parseString(s);
    } else {
      nextToken = null;
    }
  }

  protected T parseString(String s) {
    return op.apply(s);
  }

  @Override
  public boolean hasNext() {
    return nextToken != null;
  }

  @Override
  public T next() {
    if (nextToken == null) {
      throw new NoSuchElementException("DelimitRegExIterator exhausted");
    }
    T token = nextToken;
    setNext();
    return token;
  }

  public Object peek() {
    return nextToken;
  }

  /**
   * Returns a factory that vends DelimitRegExIterators that reads the contents of the
   * given Reader, splits on the specified delimiter, then returns the result.
   */
  public static IteratorFromReaderFactory<String> getFactory(String delim) {
    return DelimitRegExIteratorFactory.defaultDelimitRegExIteratorFactory(delim);
  }

  /**
   * Returns a factory that vends DelimitRegExIterators that reads the contents of the
   * given Reader, splits on the specified delimiter, then returns the result.
   */
  public static IteratorFromReaderFactory<String> getFactory(String delim, boolean eolIsSignificant) {
    return DelimitRegExIteratorFactory.defaultDelimitRegExIteratorFactory(delim, eolIsSignificant);
  }

  /**
   * Returns a factory that vends DelimitRegExIterators that reads the contents of the
   * given Reader, splits on the specified delimiter, applies op, then returns the result.
   */
  public static <T> IteratorFromReaderFactory<T> getFactory(String delim, Function<String,T> op) {
    return new DelimitRegExIteratorFactory<T>(delim, op);
  }

  /**
   * Returns a factory that vends DelimitRegExIterators that reads the contents of the
   * given Reader, splits on the specified delimiter, applies op, then returns the result.
   */
  public static <T> IteratorFromReaderFactory<T> getFactory(String delim, Function<String,T> op, boolean eolIsSignificant) {
    return new DelimitRegExIteratorFactory<T>(delim, op, eolIsSignificant);
  }

  public static class DelimitRegExIteratorFactory<T> implements IteratorFromReaderFactory<T>, Serializable {

    private static final long serialVersionUID = 6846060575832573082L;

    private String delim;
    private Function<String,T> op;
    private boolean eolIsSignificant;

    public static DelimitRegExIteratorFactory<String> defaultDelimitRegExIteratorFactory(String delim) {
      return new DelimitRegExIteratorFactory<String>(delim, new IdentityFunction<String>());
    }

    public static DelimitRegExIteratorFactory<String> defaultDelimitRegExIteratorFactory(String delim, boolean eolIsSignificant) {
      return new DelimitRegExIteratorFactory<String>(delim, new IdentityFunction<String>(), eolIsSignificant);
    }

    public DelimitRegExIteratorFactory(String delim, Function<String,T> op) {
      this(delim, op, true);
    }

    public DelimitRegExIteratorFactory(String delim, Function<String,T> op, boolean eolIsSignificant) {
      this.delim = delim;
      this.op = op;
      this.eolIsSignificant = eolIsSignificant;
    }

    public Iterator<T> getIterator(Reader r) {
      return new DelimitRegExIterator<T>(r, delim, op);
    }

  }

  public static void main(String[] args) {

    String s = "@@123\nthis\nis\na\nsentence\n\n@@124\nThis\nis\nanother\n.\n\n@125\nThis\nis\nthe\nlast\n";
    DelimitRegExIterator<String> di = DelimitRegExIterator.defaultDelimitRegExIterator(new StringReader(s), "\n\n");
    while (di.hasNext()) {
      System.out.println("****\n" + di.next() + "\n****");
    }

  }

}
