package edu.stanford.nlp.trees;

import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.objectbank.TokenizerFactory;
import edu.stanford.nlp.ling.HasIndex;

import java.io.*;
import java.util.*;


/**
 * A <code>PennTreeReader</code> is a <code>TreeReader</code> that
 * reads in Penn Treebank-style files.  Example usage:
 * <br>
 * <code> TreeReader tr = new PennTreeReader(new BufferedReader(new
 * InputStreamReader(new FileInputStream(file),"UTF-8")),
 * myTreeFactory); </code>
 *
 * @author Christopher Manning
 * @author Roger Levy
 * @version 2003/01
 */
public class PennTreeReader implements TreeReader {

  private Reader in;
  private Tokenizer<String> st;
  private TreeNormalizer tn;
  private TreeFactory tf;

  private static final boolean DEBUG = false;


  /**
   * Read parse trees from a <code>Reader</code>.
   * For the defaulted arguments, you get a
   * <code>SimpleTreeFactory</code>, no <code>TreeNormalizer</code>, and
   * a <code>PennTreebankTokenizer</code>.
   *
   * @param in The <code>Reader</code>
   */
  public PennTreeReader(Reader in) {
    this(in, new SimpleTreeFactory());
  }


  /**
   * Read parse trees from a <code>Reader</code>.
   *
   * @param in the Reader
   * @param tf TreeFactory -- factory to create some kind of Tree
   */
  public PennTreeReader(Reader in, TreeFactory tf) {
    this(in, tf, null, new PennTreebankTokenizer(in));
  }

  /**
   * Read parse trees from a <code>Reader</code>.
   *
   * @param in The Reader
   * @param st The Tokenizer
   */
  public PennTreeReader(Reader in, Tokenizer<String> st) {
    this(in, new SimpleTreeFactory(), null, st);
  }


  /**
   * Read parse trees from a Reader.
   *
   * @param in Reader
   * @param tf TreeFactory -- factory to create some kind of Tree
   * @param tn the method of normalizing trees
   */
  public PennTreeReader(Reader in, TreeFactory tf, TreeNormalizer tn) {
    this(in, tf, tn, new PennTreebankTokenizer(in));
  }


  /**
   * Read parse trees from a Reader.
   *
   * @param in Reader
   * @param tf TreeFactory -- factory to create some kind of Tree
   * @param tn the method of normalizing trees
   * @param st Tokenizer that divides up Reader
   */
  public PennTreeReader(Reader in, TreeFactory tf, TreeNormalizer tn, Tokenizer<String> st) {
    this.in = in;
    this.tf = tf;
    this.tn = tn;
    this.st = st;
    // check for whacked out headers still present in Brown corpus in Treebank 3
    String first = st.peek();
    if (first != null && first.startsWith("*x*x*x")) {
      if (DEBUG) {
        System.err.println("PennTreeReader: skipping past whacked out header.");
      }
      int foundCount = 0;
      while (foundCount < 4 && st.hasNext()) {
        first = st.next();
        if (first != null && first.startsWith("*x*x*x")) {
          foundCount++;
        }
      }
    }

    if (DEBUG) {
      System.err.println("Built PennTreeReader from " + in.getClass().getName() + " " + ((tf == null) ? "no tf" : tf.getClass().getName()) + " " + ((tn == null) ? "no tn" : tn.getClass().getName()) + " " + ((st == null) ? "no st" : st.getClass().getName()));
    }
  }

  private int wordIndex;

  /**
   * Reads a single tree in standard Penn Treebank format,
   * with or without an additional set of parens around it (an unnamed
   * ROOT node).  If the token stream ends before the current tree is complete, a
   * {@link java.util.NoSuchElementException} will get thrown from
   * deep within the innards of this method.
   *
   * @return A single tree, or <code>null</code> at end of token stream.
   */
  public Tree readTree() throws IOException {
    Tree tr = null;
    while (tr == null) {
      if (!st.hasNext()) {
        return null;
      }
      tr = readTreeHelper();
      if (DEBUG) {
        if (tr == null) { System.err.println("readTreeHelper returned null tree; continuing."); }
      }
    }
    return tr;
  }

  private Tree readTreeHelper() throws IOException {
    wordIndex = 0;
    Tree tr = readTree(st.next());
    if (tr == null || tn == null) {
      return tr;
    } else {
      return tn.normalizeWholeTree(tr, tf);
    }
  }

  private Tree readTree(String token) throws IOException {
    if (DEBUG) {
      System.out.println("readTree() next token " + token);
    }
    String name;
    // a paren starts new tree, a string is a leaf symbol,
    // o.w. IO exception
    if (token == null) {
      return null;
    } else if (token.equals(")")) {
      System.err.println("Expecting start of tree; found surplus close parenthesis ')'. Ignoring it.");
      return null;
    } else if (token.equals("(")) {
      // looks at next
      name = st.peek();
      if (DEBUG) {
        System.out.println("  peeked is \"" + name+ '\"');
      }
      // checks if it's a normal string and returns it as the label
      if (name.equals("(") || name.equals(")")) {
        name = null;
      } else {
        // get it for real
        name = st.next();
      }
      if (tn != null) {
        name = tn.normalizeNonterminal(name); // we used to .intern();
      }
      return tf.newTreeNode(name, readTrees());
    } else {
      if (tn != null) {
        name = tn.normalizeTerminal(token); // was: .intern();
      } else {
        name = token;
      }
      Tree leaf = tf.newLeaf(name);
      if (leaf.label() instanceof HasIndex) {
        HasIndex hi = (HasIndex) leaf.label();
        hi.setIndex(wordIndex);
      }
      wordIndex++;
      return leaf;
    }
  }


  /**
   * Parse sequence of trees, followed by a single right paren.
   */
  private List<Tree> readTrees() throws IOException {
    // allocate array list for temporarily storing trees
    List<Tree> parseTrees = new ArrayList<Tree>();
    // until a paren closes all subtrees, keep reading trees
    String nextToken = null;
    String fullToken = "";
    while (st.hasNext()) {
      nextToken = st.next();
      if (nextToken.equals(")")) {
        break;
      }
      else if (nextToken.equals("(")) {
	if (!fullToken.equals("")) {
	  parseTrees.add(readTree(fullToken));
	  fullToken = "";
	}
        parseTrees.add(readTree(nextToken));
      }
      else {
        fullToken += (fullToken.equals("") ? "" : " ") + nextToken;
      }
    }
    if (! ")".equals(nextToken)) {
      throw(new IOException("Expecting right paren found eof"));
    }
    if (!fullToken.equals("")) {
      parseTrees.add(readTree(fullToken));
    }

    return parseTrees;
  }


  /**
   * Close the Reader behind this <code>TreeReader</code>.
   */
  public void close() throws IOException {
    in.close();
  }

  public static TokenizerFactory<Tree> tokenizerFactory(final TreeFactory tf, final TreeNormalizer tn, final Tokenizer<String> stringTokenizer) {
    return new TreeTokenizerFactory(new TreeReaderFactory() {
      public TreeReader newTreeReader(Reader in) {
        return new PennTreeReader(in,tf,tn,stringTokenizer);
      }
    });
  }
  /*
  private static class TreeTokenizerFactory implements TokenizerFactory<Tree> {
    TreeFactory tf;
    TreeNormalizer tn;
    Tokenizer t;

    public TreeTokenizerFactory(TreeFactory tf, TreeNormalizer tn, Tokenizer t) {
      this.tf = tf;
      this.tn = tn;
      this.t = t;
    }

    public Tokenizer<Tree> getTokenizer(final Reader r) {
      return new AbstractTokenizer<Tree>() {
        PennTreeReader tr = new PennTreeReader(r,tf,tn,t);
        public Tree getNext() {
          try {
            return tr.readTree();
          }
          catch(IOException e) {
            System.err.println("Error in reading tree.");
            return null;
          }
        }
      };
    }

    public Iterator<Tree> getIterator(Reader r) {
      return getTokenizer(r);
    }
  }
  */


  /**
   * Returns an iterator over Trees which is backed by this PennTreeReader.
   * Warning: any IOExceptions which would normally be thrown are turned
   * into RuntimeExceptions.
   */
  public Iterator<Tree> asTreeIterator() {
    return new Iterator<Tree>() {
      private Tree next = advance();
      public boolean hasNext() {
        return next != null;
      }
      public Tree next() {
        if (next == null) {
          throw new NoSuchElementException("PennTreeReader exhausted");
        }
        Tree t = next;
        next = advance();
        return t;
      }
      public void remove() {
        throw new UnsupportedOperationException();
      }
      private Tree advance() {
        Tree t = readTreeThrowRuntime();
        if (t == null) closeThrowRuntime();
        return t;
      }
    };
  }

  private Tree readTreeThrowRuntime() {
    Tree t;
    try {
      t = readTree();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return t;
  }

  private void closeThrowRuntime() {
    try {
      close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Loads treebank data from first argument and prints it.
   *
   * @param args Array of command-line arguments: specifies a filename
   */
  public static void main(String[] args) {
    try {
      TreeFactory tf = new LabeledScoredTreeFactory();
      Reader r = new BufferedReader(new InputStreamReader(new FileInputStream(args[0]), "UTF-8"));
      TreeReader tr = new PennTreeReader(r, tf);
      Tree t = tr.readTree();
      while (t != null) {
        System.out.println(t);
        System.out.println();
        t = tr.readTree();
      }
      r.close();
    } catch (IOException ioe) {
      ioe.printStackTrace();
    }
  }

}
