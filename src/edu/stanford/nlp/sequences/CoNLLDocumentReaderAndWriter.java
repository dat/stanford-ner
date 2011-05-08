package edu.stanford.nlp.sequences;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.AnswerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.ChunkAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.OriginalAnswerAnnotation;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.util.PaddedList;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.AbstractIterator;

import java.util.*;
import java.util.regex.*;
import java.io.*;

/**
 * DocumentReader for CoNLL 03 format.  In this format, there is one word
 * per line, with extra attributes of a word (POS tag, chunk, etc.) in other
 * space or tab separated columns, where leading and trailing whitespace on
 * the line are ignored.  Sentences are supposedly
 * separated by a blank line (one with no non-whitespace characters), but
 * where blank lines occur is in practice often fairly random. In particular,
 * entities not infrequently span blank lines.
 *
 * @author Jenny Finkel
 * @author Huy Nguyen
 * @author Christopher Manning
 */
public class CoNLLDocumentReaderAndWriter implements DocumentReaderAndWriter {

  private static final long serialVersionUID = 6281374154299530460L;

  public static final String BOUNDARY = "*BOUNDARY*";
  public static final String OTHER = "O";
  private SeqClassifierFlags flags; // = null;


  public void init(SeqClassifierFlags flags) {
    this.flags = flags;
  }

  @Override
  public String toString() {
    return "CoNLLDocumentReaderAndWriter[entitySubclassification: " +
        flags.entitySubclassification + ", intern: " + flags.intern + ']';
  }


  public Iterator<List<CoreLabel>> getIterator(Reader r) {
    return new CoNLLIterator(r);
  }

  private class CoNLLIterator extends AbstractIterator<List<CoreLabel>> {

    public CoNLLIterator (Reader r) {
      stringIter = splitIntoDocs(r);
    }

    @Override
    public boolean hasNext() { return stringIter.hasNext(); }

    @Override
    public List<CoreLabel> next() { return processDocument(stringIter.next()); }

    private Iterator<String> stringIter; // = null;
  }

  private static Iterator<String> splitIntoDocs(Reader r) {
    return Collections.singleton(StringUtils.slurpReader(r)).iterator();
  }

  private static Pattern white = Pattern.compile("^\\s*$");

  private List<CoreLabel> processDocument(String doc) {
    List<CoreLabel> lis = new ArrayList<CoreLabel>();
    String[] lines = doc.split("\n");
    for (String line : lines) {
      if ( ! flags.deleteBlankLines || ! white.matcher(line).matches()) {
        lis.add(makeCoreLabel(line));
      }
    }
    entitySubclassify(lis, flags.entitySubclassification);
    return lis;
  }

  /**
   * This was used on the CoNLL data to map from a representation where
   * normally entities were marked I-PERS, but the beginning of non-first
   * items of an entity sequences were marked B-PERS (IOB1 representation).
   * It changes this representation to other representations:
   * a 4 way representation of all entities, like S-PERS, B-PERS,
   * I-PERS, E-PERS for single word, beginning, internal, and end of entity
   * (SBIEO); always marking the first word of an entity (IOB2);
   * the reverse IOE1 and IOE2 and IO.
   * This code is very specific to the particular CoNLL way of labeling
   * classes.  It will work on any of these styles of input, however, except
   * for IO which necessarily loses information.
   */
  private void entitySubclassify(List<CoreLabel> lineInfos,
                                 String style) {
    int how;
    if ("iob1".equalsIgnoreCase(style)) {
      how = 0;
    } else if ("iob2".equalsIgnoreCase(style)) {
      how = 1;
    } else if ("ioe1".equalsIgnoreCase(style)) {
      how = 2;
    } else if ("ioe2".equalsIgnoreCase(style)) {
      how = 3;
    } else if ("io".equalsIgnoreCase(style)) {
      how = 4;
    } else if ("sbieo".equalsIgnoreCase(style)) {
      how = 5;
    } else {
      System.err.println("entitySubclassify: unknown style: " + style);
      how = 4;
    }
    lineInfos = new PaddedList<CoreLabel>(lineInfos, new CoreLabel());
    int k = lineInfos.size();
    String[] newAnswers = new String[k];
    for (int i = 0; i < k; i++) {
      final CoreLabel c = lineInfos.get(i);
      final CoreLabel p = lineInfos.get(i - 1);
      final CoreLabel n = lineInfos.get(i + 1);
      final String cAns = c.get(AnswerAnnotation.class);
      if (cAns.length() > 1 && cAns.charAt(1) == '-') {
        String pAns = p.get(AnswerAnnotation.class);
        if (pAns == null) { pAns = OTHER; }
        String nAns = n.get(AnswerAnnotation.class);
        if (nAns == null) { nAns = OTHER; }
        final String base = cAns.substring(2, cAns.length());
        String pBase = (pAns.length() > 2 ? pAns.substring(2, pAns.length()) : pAns);
        String nBase = (nAns.length() > 2 ? nAns.substring(2, nAns.length()) : nAns);
        char prefix = cAns.charAt(0);
        char pPrefix = (pAns.length() > 0) ? pAns.charAt(0) : ' ';
        char nPrefix = (nAns.length() > 0) ? nAns.charAt(0) : ' ';
        boolean isStartAdjacentSame = base.equals(pBase) &&
          (prefix == 'B' || prefix == 'S' || pPrefix == 'E' || pPrefix == 'S');
        boolean isEndAdjacentSame = base.equals(nBase) &&
          (prefix == 'E' || prefix == 'S' || nPrefix == 'B' || pPrefix == 'S');
        boolean isFirst = (!base.equals(pBase)) || cAns.charAt(0) == 'B';
        boolean isLast = (!base.equals(nBase)) || nAns.charAt(0) == 'B';
        switch (how) {
        case 0:
          if (isStartAdjacentSame) {
            newAnswers[i] = intern("B-" + base);
          } else {
            newAnswers[i] = intern("I-" + base);
          }
          break;
        case 1:
          if (isFirst) {
            newAnswers[i] = intern("B-" + base);
          } else {
            newAnswers[i] = intern("I-" + base);
          }
          break;
        case 2:
          if (isEndAdjacentSame) {
            newAnswers[i] = intern("E-" + base);
          } else {
            newAnswers[i] = intern("I-" + base);
          }
          break;
        case 3:
          if (isLast) {
            newAnswers[i] = intern("E-" + base);
          } else {
            newAnswers[i] = intern("I-" + base);
          }
          break;
        case 4:
          newAnswers[i] = intern("I-" + base);
          break;
        case 5:
          if (isFirst && isLast) {
            newAnswers[i] = intern("S-" + base);
          } else if ((!isFirst) && isLast) {
            newAnswers[i] = intern("E-" + base);
          } else if (isFirst && (!isLast)) {
            newAnswers[i] = intern("B-" + base);
          } else {
            newAnswers[i] = intern("I-" + base);
          }
        }
      } else {
        newAnswers[i] = cAns;
      }
    }
    for (int i = 0; i < k; i++) {
      CoreLabel c = lineInfos.get(i);
      c.set(AnswerAnnotation.class, newAnswers[i]);
    }
  }

  private CoreLabel makeCoreLabel(String line) {
    CoreLabel wi = new CoreLabel();
    // wi.line = line;
    String[] bits = line.split("\\s+");
    switch (bits.length) {
    case 0:
    case 1:
      wi.setWord(BOUNDARY);
      wi.set(AnswerAnnotation.class, OTHER);
      break;
    case 2:
      wi.setWord(bits[0]);
      wi.set(AnswerAnnotation.class, bits[1]);
      break;
    case 3:
      wi.setWord(bits[0]);
      wi.setTag(bits[1]);
      wi.set(AnswerAnnotation.class, bits[2]);
      break;
    case 4:
      wi.setWord(bits[0]);
      wi.setTag(bits[1]);
      wi.set(ChunkAnnotation.class, bits[2]);
      wi.set(AnswerAnnotation.class, bits[3]);
      break;
    case 5:
      if (flags.useLemmaAsWord) {
        wi.setWord(bits[1]);
      } else {
        wi.setWord(bits[0]);
        }
      wi.set(LemmaAnnotation.class, bits[1]);
      wi.setTag(bits[2]);
      wi.set(ChunkAnnotation.class, bits[3]);
      wi.set(AnswerAnnotation.class, bits[4]);
      break;
    default:
      throw new RuntimeIOException("Unexpected input (many fields): " + line);
    }
    wi.set(OriginalAnswerAnnotation.class, wi.get(AnswerAnnotation.class));
    // This collapses things to do neither iob1 or iob2 but just IO. Remove?
    // if (wi.get(AnswerAnnotation.class).length() > 1 && wi.get(AnswerAnnotation.class).charAt(1) == '-' && !flags.useFourWayEntitySubclassification) {
    //  wi.set(AnswerAnnotation.class, "I-" + wi.get(AnswerAnnotation.class).substring(2));
    // }
    return wi;
  }

  private String intern(String s) {
    if (flags.intern) {
      return s.intern();
    } else {
      return s;
    }
  }

  /** Return the marking scheme to IOB1 marking, regardless of what it was.
   *  @param lineInfos List of tokens in some NER encoding
   */
  private void deEndify(List<CoreLabel> lineInfos) {
    if (flags.retainEntitySubclassification) {
      return;
    }
    lineInfos = new PaddedList<CoreLabel>(lineInfos, new CoreLabel());
    int k = lineInfos.size();
    String[] newAnswers = new String[k];
    for (int i = 0; i < k; i++) {
      CoreLabel c = lineInfos.get(i);
      CoreLabel p = lineInfos.get(i - 1);
      if (c.get(AnswerAnnotation.class).length() > 1 && c.get(AnswerAnnotation.class).charAt(1) == '-') {
        String base = c.get(AnswerAnnotation.class).substring(2);
        String pBase = (p.get(AnswerAnnotation.class).length() <= 2 ? p.get(AnswerAnnotation.class) : p.get(AnswerAnnotation.class).substring(2));
        boolean isSecond = (base.equals(pBase));
        boolean isStart = (c.get(AnswerAnnotation.class).charAt(0) == 'B' || c.get(AnswerAnnotation.class).charAt(0) == 'S');
        if (isSecond && isStart) {
          newAnswers[i] = intern("B-" + base);
        } else {
          newAnswers[i] = intern("I-" + base);
        }
      } else {
        newAnswers[i] = c.get(AnswerAnnotation.class);
      }
    }
    for (int i = 0; i < k; i++) {
      CoreLabel c = lineInfos.get(i);
      c.set(AnswerAnnotation.class, newAnswers[i]);
    }
  }


  /**
   * @param doc The document: A List of CoreLabel
   * @param out Where to send the answers to
   */
  public void printAnswers(List<CoreLabel> doc, PrintWriter out) {
    // boolean tagsMerged = flags.mergeTags;
    // boolean useHead = flags.splitOnHead;

    if ( ! "iob1".equalsIgnoreCase(flags.entitySubclassification)) {
      deEndify(doc);
    }
    String prevGold = "";
    String prevGuess = "";

    for (CoreLabel fl : doc) {
      String word = fl.word();
      if (word == BOUNDARY) {
        out.println();
      } else {
        String gold = fl.get(OriginalAnswerAnnotation.class);
        if(gold == null) gold = "";
        String guess = fl.get(AnswerAnnotation.class);
        // System.err.println(fl.word() + "\t" + fl.goldget(AnswerAnnotation.class) + "\t" + fl.get(AnswerAnnotation.class));
        if (false) {
          // chris aug 2005
          // this bit of code was here, and it appears like it would
          // always mark the first of an entity sequence as B-, i.e.,
          // IOB2, but CoNLL uses IOB1, which only marks with B- when two
          // entities are adjacent, an annotation we just lose on.
          // now just record unmucked with origAnswer so can't need to do this
          if ( ! gold.equals(OTHER) && gold.length() >= 2) {
            if ( ! gold.substring(2).equals(prevGold)) {
              gold = "B-" + gold.substring(2);
            }
            prevGold = gold.substring(2);
          }
          if ( ! guess.equals(OTHER) && guess.length() >= 2) {
            if ( ! guess.substring(2).equals(prevGuess)) {
              guess = "B-" + guess.substring(2);
            }
            prevGuess = guess;
          }
        }
        String pos = fl.tag();
        String chunk = (fl.get(ChunkAnnotation.class) == null ? "" : fl.get(ChunkAnnotation.class));
        out.println(fl.word() + '\t' + pos + '\t' + chunk + '\t' +
                    gold + '\t' + guess);
      }
    }
  }

  /** Count some stats on what occurs in a file.
   */
  public static void main(String[] args) throws IOException, ClassNotFoundException {
//     CoNLLDocumentReaderAndWriter f = new CoNLLDocumentReaderAndWriter();
//     int numTokens = 0;
//     int numEntities = 0;
//     String lastAnsBase = "";
//     List<CoreLabel> ll = f.processDocument(args[0]);
//     for (CoreLabel fl : ll) {
//       // System.out.println("FL " + (++i) + " was " + fl);
//       if (fl.word().equals(BOUNDARY)) {
//         continue;
//       }
//       String ans = fl.get(AnswerAnnotation.class);
//       String ansBase;
//       String ansPrefix;
//       String[] bits = ans.split("-");
//       if (bits.length == 1) {
//         ansBase = bits[0];
//         ansPrefix = "";
//       } else {
//         ansBase = bits[1];
//         ansPrefix = bits[0];
//       }
//       numTokens++;
//       if (ansBase.equals("O")) {
//       } else if (ansBase.equals(lastAnsBase)) {
//         if (ansPrefix.equals("B")) {
//           numEntities++;
//         }
//       } else {
//         numEntities++;
//       }
//     }
//     System.out.println("File " + args[0] + " has " + numTokens +
//                        " tokens and " + numEntities + " entities.");
  } // end main

} // end class CoNLLDocumentReaderAndWriter
