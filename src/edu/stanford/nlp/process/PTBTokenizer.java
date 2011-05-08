package edu.stanford.nlp.process;


import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.objectbank.TokenizerFactory;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;
import edu.stanford.nlp.io.IOUtils;


/**
 * Tokenizer implementation that conforms to the Penn Treebank tokenization
 * conventions.
 * This tokenizer is a Java implementation of Professor Chris Manning's Flex
 * tokenizer, pgtt-treebank.l.  It reads raw text and outputs
 * tokens as edu.stanford.nlp.trees.Words in the Penn treebank format. It can
 * optionally return carriage returns as tokens.
 *
 * @author Tim Grow
 * @author Teg Grenager (grenager@stanford.edu)
 * @author Christopher Manning
 * @author Jenny Finkel (integrating in invertible PTB tokenizer)
 */
public class PTBTokenizer<T extends HasWord> extends AbstractTokenizer<T> {

  // todo: clean up treatment of invertible. Make it less of a type-changing parameter (from Word to CoreLabel)
  // todo: let Americanization be able to be turned off separately of other PTB escaping
  // todo: have the various options available to clients

  // whether carriage returns should be returned as tokens
  private boolean tokenizeCRs;
  private boolean invertible;
  private boolean suppressEscaping; // = false;

  // the underlying lexer
  private PTBLexer lexer;
  private LexedTokenFactory<T> tokenFactory;
  // private int position;

  /**
   * Constructs a new PTBTokenizer that treats carriage returns as normal
   * whitespace.
   *
   * @param r The Reader whose contents will be tokenized
   * @return A PTBTokenizer that tokenizes a stream to objects of type
   *          {@link Word}
   */
  public static PTBTokenizer<Word> newPTBTokenizer(Reader r) {
    return newPTBTokenizer(r, false);
  }

  /**
   * Constructs a new PTBTokenizer that optionally returns carriage returns
   * as their own token. CRs come back as Words whose text is
   * the value of <code>PTBLexer.cr</code>.
   *
   * @param r The Reader to read tokens from
   * @param tokenizeCRs Whether to return newlines as separate tokens
   *         (otherwise they normally disappear as whitespace)
   * @return A PTBTokenizer which returns Word tokens
   */
  public static PTBTokenizer<Word> newPTBTokenizer(Reader r, boolean tokenizeCRs) {
    return new PTBTokenizer<Word>(r, tokenizeCRs, new WordTokenFactory());
  }


  /**
   * Constructs a new PTBTokenizer that optionally returns carriage returns
   * as their own token. CRs come back as Words whose text is
   * the value of <code>PTBLexer.cr</code>.
   *
   * @param r The Reader to read tokens from
   * @param tokenizeCRs Whether to return newlines as separate tokens
   *         (otherwise they normally disappear as whitespace)
   * @param invertible if set to true, then will produce CoreLabels which
   *         will have fields for the string before and after, and the
   *         character offsets
   * @return A PTBTokenizer which returns CoreLabel objects
   */
  public static PTBTokenizer<CoreLabel> newPTBTokenizer(Reader r, boolean tokenizeCRs, boolean invertible) {
    return new PTBTokenizer<CoreLabel>(r, tokenizeCRs, invertible, new CoreLabelTokenFactory());
  }


  /**
   * Constructs a new PTBTokenizer that optionally returns carriage returns
   * as their own token, and has a custom LexedTokenFactory.
   * CRs come back as Words whose text is
   * the value of <code>PTBLexer.cr</code>.
   *
   * @param r The Reader to read tokens from
   * @param tokenizeCRs Whether to return newlines as separate tokens
   *         (otherwise they normally disappear as whitespace)
   * @param tokenFactory The LexedTokenFactory to use to create
   *         tokens from the text.
   */
  public PTBTokenizer(Reader r, boolean tokenizeCRs,
      LexedTokenFactory<T> tokenFactory) {
    this (r, tokenizeCRs, false, tokenFactory);
  }

  private PTBTokenizer(Reader r, boolean tokenizeCRs, boolean invertible,
                       LexedTokenFactory<T> tokenFactory) {
    this(r, tokenizeCRs, invertible, false, tokenFactory);
  }

  private PTBTokenizer(Reader r, boolean tokenizeCRs, boolean invertible,
                       boolean suppressEscaping,
                       LexedTokenFactory<T> tokenFactory) {
    this.tokenizeCRs = tokenizeCRs;
    this.tokenFactory = tokenFactory;
    this.invertible = invertible;
    this.suppressEscaping = suppressEscaping;
    setSource(r);
  }


  /**
   * Internally fetches the next token.
   *
   * @return the next token in the token stream, or null if none exists.
   */
  @Override
  @SuppressWarnings("unchecked")
  protected T getNext() {
    // if (lexer == null) {
    //   return null;
    // }
    T token = null;
    try {
      token = (T) lexer.next();
      // cdm 2007: this shouldn't be necessary: PTBLexer decides for itself whether to return CRs based on the same flag!
      // get rid of CRs if necessary
      // while (!tokenizeCRs && PTBLexer.cr.equals(((HasWord) token).word())) {
      //   token = (T)lexer.next();
      // }
    } catch (Exception e) {
      nextToken = null;
      // do nothing, return null
    }
    return token;
  }

  /**
   * Sets the source of this Tokenizer to be the Reader r.
   * @param r The Reader to tokenize from
   */
  public final void setSource(Reader r) {
    if (invertible) {
      lexer = new PTBLexer(r, invertible, tokenizeCRs);
    } else {
      lexer = new PTBLexer(r, tokenFactory, tokenizeCRs, suppressEscaping);
    }
    // position = 0;
  }

  /**
   * Returns a presentable version of the given PTB-tokenized text.
   * PTB tokenization splits up punctuation and does various other things
   * that makes simply joining the tokens with spaces look bad. So join
   * the tokens with space and run it through this method to produce nice
   * looking text. It's not perfect, but it works pretty well.
   */
  public static String ptb2Text(String ptbText) {
    StringBuilder sb = new StringBuilder(ptbText.length()); // probably an overestimate
    PTB2TextLexer lexer = new PTB2TextLexer(new StringReader(ptbText));
    try {
      for (String token; (token = lexer.next()) != null; ) {
        sb.append(token);
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
    return (sb.toString());
  }

  /**
   * Returns a presentable version of the given PTB-tokenized text.
   * PTB tokenization splits up punctuation and does various other things
   * that makes simply joining the tokens with spaces look bad. So join
   * the tokens with space and run it through this method to produce nice
   * looking text. It's not perfect, but it works pretty well.
   */
  public static int ptb2Text(Reader ptbText, Writer w) throws IOException {
    int numTokens = 0;
    PTB2TextLexer lexer = new PTB2TextLexer(ptbText);
    for (String token; (token = lexer.next()) != null; ) {
      numTokens++;
      w.write(token);
    }
    return numTokens;
  }

  private static void untok(List<String> inputFileList, List<String> outputFileList, String charset) throws IOException {
    Timing t = new Timing();
    int numTokens = 0;
    int sz = inputFileList.size();
    if (sz == 0) {
      Reader r = new InputStreamReader(System.in, charset);
      PrintWriter out = new PrintWriter(System.out, true);
      numTokens = ptb2Text(r, out);
    } else {
      for (int j = 0; j < sz; j++) {
        Reader r = IOUtils.readReaderFromString(inputFileList.get(j), charset);
        PrintWriter out;
        if (outputFileList == null) {
          out = new PrintWriter(System.out, true);
        } else {
          out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFileList.get(j)), charset)), true);
        }
        numTokens += ptb2Text(r, out);
        out.close();
      }
    }
    long millis = t.stop();
    double wordspersec = numTokens / (((double) millis) / 1000);
    NumberFormat nf = new DecimalFormat("0.00"); // easier way!
    System.err.println("PTBTokenizer untokenized " + numTokens + " tokens at " +
                       nf.format(wordspersec) + " tokens per second.");
  }

  /**
   * Returns a presentable version of the given PTB-tokenized words.
   * Pass in a List of Strings and this method will
   * join the words with spaces and call {@link #ptb2Text(String)} on the
   * output.
   */
  public static String ptb2Text(List<String> ptbWords) {
    return ptb2Text(StringUtils.join(ptbWords));
  }


  /**
   * Returns a presentable version of the given PTB-tokenized words.
   * Pass in a List of Words or a Document and this method will
   * join the words with spaces and call {@link #ptb2Text(String)} on the
   * output. This method will take the word() values to prevent additional
   * text from creeping in (e.g., POS tags).
   */
  public static String labelList2Text(List<? extends HasWord> ptbWords) {
    List<String> words = new ArrayList<String>();
    for (HasWord hw : ptbWords) {
      words.add(hw.word());
    }

    return ptb2Text(words);
  }


  private static void tok(List<String> inputFileList, List<String> outputFileList, String charset, Pattern parseInsideBegin, Pattern parseInsideEnd, boolean tokenizeNL, boolean preserveLines, boolean dump) throws IOException {
    Timing t = new Timing();
    int numTokens = 0;
    int sz = inputFileList.size();
    if (sz == 0) {
      Reader r = new InputStreamReader(System.in, charset);
      PrintWriter out = new PrintWriter(System.out, true);
      numTokens += tokReader(r, out, parseInsideBegin, parseInsideEnd, tokenizeNL, preserveLines, dump);
    } else {
      for (int j = 0; j < sz; j++) {
        Reader r = IOUtils.readReaderFromString(inputFileList.get(j), charset);
        PrintWriter out;
        if (outputFileList == null) {
          out = new PrintWriter(System.out, true);
        } else {
          out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFileList.get(j)), charset)), true);
        }

        numTokens += tokReader(r, out, parseInsideBegin, parseInsideEnd, tokenizeNL, preserveLines, dump);
        r.close();
        if (outputFileList != null) out.close();
      } // end for j going through inputFileList
    }
    long millis = t.stop();
    double wordspersec = numTokens / (((double) millis) / 1000);
    NumberFormat nf = new DecimalFormat("0.00"); // easier way!
    System.err.println("PTBTokenizer tokenized " + numTokens + " tokens at " +
                       nf.format(wordspersec) + " tokens per second.");
  }

  private static int tokReader(Reader r, PrintWriter out, Pattern parseInsideBegin, Pattern parseInsideEnd, boolean tokenizeNL, boolean preserveLines, boolean dump) {
    int numTokens = 0;
    PTBTokenizer<CoreLabel> tokenizer = PTBTokenizer.newPTBTokenizer(r, tokenizeNL, true);
    boolean printing = parseInsideBegin == null; // start off printing, unless you're looking for a start entity
    boolean beginLine = true;
    while (tokenizer.hasNext()) {
      CoreLabel obj = tokenizer.next();
      String str = obj.word();

      if (parseInsideBegin != null && parseInsideBegin.matcher(str).matches()) {
        printing = true;
      } else if (parseInsideEnd != null && parseInsideEnd.matcher(str).matches()) {
        printing = false;
      } else if (printing) {
        if (dump) {
          // after having checked for tags, change str to be exhaustive
          str = obj.toString();
        }
        if (preserveLines) {
          if ("*CR*".equals(str)) {
            beginLine = true;
            out.println();
          } else {
            if ( ! beginLine) {
              out.print(" ");
            } else {
              beginLine = false;
            }
            out.print(str);
          }
        } else {
          out.println(str);
        }
      }
      numTokens++;
    }
    return numTokens;
  }


  public static TokenizerFactory<Word> factory() {
    return PTBTokenizerFactory.newPTBTokenizerFactory();
  }

  public static TokenizerFactory<Word> factory(boolean tokenizeCRs) {
    return PTBTokenizerFactory.newPTBTokenizerFactory(tokenizeCRs);
  }


  public static <T extends HasWord> TokenizerFactory<T> factory(boolean tokenizeCRs, LexedTokenFactory<T> factory) {
    return new PTBTokenizerFactory<T>(tokenizeCRs, factory);
  }

  public static TokenizerFactory<CoreLabel> factory(boolean tokenizeCRs, boolean invertible) {
    return PTBTokenizerFactory.newPTBTokenizerFactory(tokenizeCRs, invertible);
  }

  public static TokenizerFactory<Word> factory(boolean tokenizeCRs, boolean invertible, boolean suppressEscaping) {
    return PTBTokenizerFactory.newPTBTokenizerFactory(tokenizeCRs, invertible, suppressEscaping);
  }


  public static class PTBTokenizerFactory<T extends HasWord> implements TokenizerFactory<T> {

    protected boolean tokenizeCRs;
    protected boolean invertible;
    protected boolean suppressEscaping; // = false;
    protected LexedTokenFactory<T> factory;

    /**
     * Constructs a new PTBTokenizerFactory that treats carriage returns as
     * normal whitespace and returns Word objects.
     *
     * @return A TokenizerFactory that returns Word objects
     */
    public static PTBTokenizerFactory<Word> newPTBTokenizerFactory() {
      return newPTBTokenizerFactory(false);
    }

    /**
     * Constructs a new PTBTokenizer that optionally returns carriage returns
     * as their own token.
     *
     * @param tokenizeCRs If true, CRs come back as Words whose text is
     *    the value of <code>PTBLexer.cr</code>.
     * @return A TokenizerFactory that returns Word objects
     */
    public static PTBTokenizerFactory<Word> newPTBTokenizerFactory(boolean tokenizeCRs) {
      return new PTBTokenizerFactory<Word>(tokenizeCRs, new WordTokenFactory());
    }

    public PTBTokenizerFactory(boolean tokenizeCRs, LexedTokenFactory<T> factory) {
      this(tokenizeCRs, false, false, factory);
    }

    public static PTBTokenizerFactory<CoreLabel> newPTBTokenizerFactory(boolean tokenizeCRs, boolean invertible) {
      return new PTBTokenizerFactory<CoreLabel>(tokenizeCRs, invertible, new CoreLabelTokenFactory());
    }

    // I'm not sure what will happen
    // if you set both invertible and suppressEscaping to true.
    // -pichuan (Wed Jan 31 23:12:04 2007)
    public static PTBTokenizerFactory<Word> newPTBTokenizerFactory(boolean tokenizeCRs, boolean invertible, boolean suppressEscaping) {
      return new PTBTokenizerFactory<Word>(tokenizeCRs, invertible, suppressEscaping, new WordTokenFactory());
    }

    private PTBTokenizerFactory(boolean tokenizeCRs, boolean invertible, LexedTokenFactory<T> factory) {
      this(tokenizeCRs, invertible, false, factory);
    }

    private PTBTokenizerFactory(boolean tokenizeCRs, boolean invertible, boolean suppressEscaping, LexedTokenFactory<T> factory) {
      this.tokenizeCRs = tokenizeCRs;
      this.invertible = invertible;
      this.suppressEscaping = suppressEscaping;
      this.factory = factory;
    }


    public Iterator<T> getIterator(Reader r) {
      return getTokenizer(r);
    }

    public Tokenizer<T> getTokenizer(Reader r) {
      return new PTBTokenizer<T>(r, tokenizeCRs, invertible, suppressEscaping, factory);
    }

  } // end static class PTBTokenizerFactory


  /**
   * Reads files named as arguments and print their tokens, by default as
   * one per line.  This is useful either for testing or to run
   * standalone to turn a corpus into a one-token-per-line file of tokens.
   * This main method assumes that the input file is in utf-8 encoding,
   * unless it is specified.
   * <p/>
   * Usage: <code>
   * java edu.stanford.nlp.process.PTBTokenizer [options] filename+
   * </code>
   * <p/>
   * Options:
   * <ul>
   * <li> -nl Tokenize newlines as tokens
   * <li> -preserveLines Produce space-separated tokens, except
   *       when the original had a line break, not one-token-per-line
   * <li> -charset charset Specifies a character encoding
   * <li> -parseInside regex Names an XML-style tag or a regular expression
   *      over such elements.  The tokenizer will only tokenize inside element
   *      that match this name.  (This is done by regex matching, not an XML
   *      parser, but works well for simply XML documents, or other SGML-style
   *      documents, such as Linguistic Data Consortium releases.)
   * <li> -ioFileList file* The remaining command-line arguments are treated as
   *      filenames that themselves contain lists of pairs of input-output
   *      filenames (2 column, whitespace separated).
   * <li> -dump Print the whole of each CoreLabel, not just the value (word)
   * <li> -untok Heuristically untokenize tokenized text
   * <li>-h Print usage info
   * </ul>
   *
   * @param args Command line arguments
   * @throws IOException If any file I/O problem
   */
  public static void main(String[] args) throws IOException {
    int i = 0;
    String charset = "utf-8";
    Pattern parseInsideBegin = null;
    Pattern parseInsideEnd = null;
    boolean tokenizeNL = false;
    boolean preserveLines = false;
    boolean inputOutputFileList = false;
    boolean dump = false;
    boolean untok = false;

    while (i < args.length && args[i].charAt(0) == '-') {
      if ("-nl".equals(args[i])) {
        tokenizeNL = true;
      } else if ("-preserveLines".equals(args[i])) {
        preserveLines = true;
        tokenizeNL = true;
      } else if ("-dump".equals(args[i])) {
        dump = true;
      } else if ("-ioFileList".equals(args[i])) {
        inputOutputFileList = true;
      } else if ("-charset".equals(args[i]) && i < args.length - 1) {
        i++;
        charset = args[i];
      } else if ("-parseInside".equals(args[i]) && i < args.length - 1) {
        i++;
        try {
          parseInsideBegin = Pattern.compile("<(?:" + args[i] + ")[^>]*?>");
          parseInsideEnd = Pattern.compile("</(?:" + args[i] + ")[^>]*?>");
        } catch (Exception e) {
          parseInsideBegin = null;
          parseInsideEnd = null;
        }
      } else if ("-untok".equals(args[i])) {
        untok = true;
      } else if ("-h".equals(args[i]) || "-help".equals(args[i]) || "--help".equals(args[i])) {
        System.err.println("usage: java edu.stanford.nlp.process.PTBTokenizer [options]* filename*");
        System.err.println("  options: -nl|-preserveLines|-dump|-ioFileList|-charset|-parseInside|-h");
        return;  // exit if they asked for help in options
      } else {
        System.err.println("Unknown option: " + args[i]);
      }
      i++;
    }

    ArrayList<String> inputFileList = new ArrayList<String>();
    ArrayList<String> outputFileList = null;

    if (inputOutputFileList) {
      outputFileList = new ArrayList<String>();
      for (int j = i; j < args.length; j++) {
        BufferedReader r = new BufferedReader(
          new InputStreamReader(new FileInputStream(args[j]), charset));
        for (String inLine; (inLine = r.readLine()) != null; ) {
          String[] fields = inLine.split("\\s+");
          inputFileList.add(fields[0]);
          if (fields.length > 1) {
            outputFileList.add(fields[1]);
          } else {
            outputFileList.add(fields[0] + ".tok");
          }
        }
        r.close();
      }
    } else {
      inputFileList.addAll(Arrays.asList(args).subList(i, args.length));
    }

    if (untok) {
      untok(inputFileList, outputFileList, charset);
    } else {
      tok(inputFileList, outputFileList, charset, parseInsideBegin, parseInsideEnd, tokenizeNL, preserveLines, dump);
    }
  } // end main

} // end PTBTokenizer
