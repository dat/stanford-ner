package edu.stanford.nlp.sequences;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.AnswerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.BeforeAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.AfterAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CurrentAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PositionAnnotation;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.WordToSentenceProcessor;
import edu.stanford.nlp.util.XMLUtils;

import java.io.*;
import java.util.*;
import java.util.regex.*;

/** This provides methods for reading plain text documents and writing out
 *  those documents once classified in several different formats.
 *
 *  @author Jenny Finkel
 *  @author Christopher Manning (new output options organization)
 */
public class PlainTextDocumentReaderAndWriter implements DocumentReaderAndWriter {

  private static final long serialVersionUID = -2420535144980273136L;

  public static final int OUTPUT_STYLE_SLASH_TAGS = 1;
  public static final int OUTPUT_STYLE_XML = 2;
  public static final int OUTPUT_STYLE_INLINE_XML = 3;

  private static final Pattern sgml = Pattern.compile("<[^>]*>");
  private static final WordToSentenceProcessor<CoreLabel,Object,Object> wts = new WordToSentenceProcessor<CoreLabel,Object,Object>();

  private SeqClassifierFlags flags; // = null;


  /** Construct a PlainTextDocumentReaderAndWriter.
   *  You should call init() after using the constructor.
   */
  public PlainTextDocumentReaderAndWriter() {
  }


  public void init(SeqClassifierFlags flags) {
    this.flags = flags;
  }


  public Iterator<List<CoreLabel>> getIterator(Reader r) {
    PTBTokenizer<CoreLabel> ptb = PTBTokenizer.newPTBTokenizer(r, false, true);
    List<CoreLabel> words = new ArrayList<CoreLabel>();
    CoreLabel previous = new CoreLabel();
    StringBuilder prepend = new StringBuilder();

    /* This changes SGML tags into whitespace -- it should maybe be moved elsewhere */
    while (ptb.hasNext()) {
      CoreLabel w = ptb.next();
      Matcher m = sgml.matcher(w.word());
      if (m.matches()) {
        prepend.append(w.before()).append(w.word());
        previous.appendAfter(w.word() + w.after());
      } else {
        if (prepend.length() > 0) {
          w.prependBefore(prepend.toString());
          prepend = new StringBuilder();
        }
        words.add(w);
        previous = w;
      }
    }

    List<List<CoreLabel>> sentences = wts.process(words);
    String after = "";
    CoreLabel last = null;
    for (List<CoreLabel> sentence : sentences) {
      int pos = 0;
      for (CoreLabel w : sentence) {
        w.set(PositionAnnotation.class, Integer.toString(pos));
        after = w.after();
        w.remove(AfterAnnotation.class);
        last = w;
      }
    }
    if (last != null) { last.set(AfterAnnotation.class, after); }

    return sentences.iterator();
  }


  /** @deprecated This has been left in since it is still called in the version
   *  of the tagger that we currently distribute, but it will be removed.
   */
  public static String getAnswers(List<CoreLabel> l) {
    StringBuilder sb = new StringBuilder();
    String after = "";
    for (CoreLabel wi : l) {
      String prev = wi.getString(BeforeAnnotation.class);
      after = wi.getString(AfterAnnotation.class);
      sb.append(prev).append(wi.word()).append('/').append(wi.getString(AnswerAnnotation.class));
    }
    sb.append(after);

    return sb.toString();
  }


  public static int asIntOutputFormat(String outputFormat) {
    if ("slashTags".equals(outputFormat)) {
      return OUTPUT_STYLE_SLASH_TAGS;
    } else if ("xml".equals(outputFormat)) {
      return OUTPUT_STYLE_XML;
    } else if ("inlineXML".equals(outputFormat)) {
      return OUTPUT_STYLE_INLINE_XML;
    } else {
      throw new IllegalArgumentException("outputFormat must be slashTags, xml, or inlineXML, not" + outputFormat);
    }
  }


  /** Print the classifications for the document to the given Writer.
   *  This method now checks the <code>outputFormat</code> property,
   *  and can print in slashTags, inlineXML, or xml (stand-Off XML).
   *  For both the XML output formats, it preserves spacing, while for the
   *  slashTags format, it prints tokenized (since preserveSpacing output is
   *  somewhat dysfunctional with the slashTags format).
   *
   *  @param list List of tokens with classifier answers
   *  @param out Where to print the output to
   */
  public void printAnswers(List<CoreLabel> list, PrintWriter out) {
    String style = null;
    if (flags != null) {
      style = flags.outputFormat;
    }
    if (style == null || "".equals(style)) {
      style = "slashTags";
    }
    int outputStyle = asIntOutputFormat(style);
    printAnswers(list, out, outputStyle, ! "slashTags".equals(style));
  }


  public String getAnswers(List<CoreLabel> l, int outputStyle, boolean preserveSpacing) {
    StringWriter sw = new StringWriter();
    PrintWriter pw = new PrintWriter(sw);
    printAnswers(l, pw, outputStyle, preserveSpacing);
    pw.flush();
    return sw.toString();
  }


  public void printAnswers(List<CoreLabel> l, PrintWriter out, int outputStyle, boolean preserveSpacing) {
    switch (outputStyle) {
      case OUTPUT_STYLE_SLASH_TAGS:
        if (preserveSpacing) {
          printAnswersAsIsText(l, out);
        } else {
          printAnswersTokenizedText(l, out);
        }
        break;
      case OUTPUT_STYLE_XML:
        if (preserveSpacing) {
          printAnswersXML(l, out);
        } else {
          printAnswersTokenizedXML(l, out);
        }
        break;
      case OUTPUT_STYLE_INLINE_XML:
        if (preserveSpacing) {
          printAnswersInlineXML(l, out);
        } else {
          printAnswersTokenizedInlineXML(l, out);
        }
        break;
      default:
        throw new IllegalArgumentException("outputStyle must be 1, 2, or 3, not" + outputStyle);
    }
  }


  private static void printAnswersTokenizedText(List<CoreLabel> l, PrintWriter out) {
    for (CoreLabel wi : l) {
      out.print(wi.word());
      out.print('/');
      out.print(wi.getString(AnswerAnnotation.class));
      out.print(' ');
    }
  }


  private static void printAnswersAsIsText(List<CoreLabel> l, PrintWriter out) {
    for (CoreLabel wi : l) {
      out.print(wi.getString(BeforeAnnotation.class));
      out.print(wi.word());
      out.print('/');
      out.print(wi.getString(AnswerAnnotation.class));
      out.print(wi.getString(AfterAnnotation.class));
    }
  }


  private static void printAnswersXML(List<CoreLabel> doc, PrintWriter out) {
    int num = 0;
    for (CoreLabel wi : doc) {
      String prev = wi.getString(BeforeAnnotation.class);
      out.print(prev);
      out.print("<wi num=\"");
      //tag.append(wi.get("position"));
      out.print(num++);
      out.print("\" entity=\"");
      out.print(wi.getString(AnswerAnnotation.class));
      out.print("\">");
      out.print(XMLUtils.escapeXML(wi.word()));
      out.print("</wi>");
      String after = wi.getString(AfterAnnotation.class);
      out.print(after);
    }
  }

  private static void printAnswersTokenizedXML(List<CoreLabel> doc, PrintWriter out) {
    int num = 0;
    for (CoreLabel wi : doc) {
      out.print("<wi num=\"");
      //tag.append(wi.get("position"));
      out.print(num++);
      out.print("\" entity=\"");
      out.print(wi.getString(AnswerAnnotation.class));
      out.print("\">");
      out.print(XMLUtils.escapeXML(wi.word()));
      out.println("</wi>");
    }
  }


  private void printAnswersInlineXML(List<CoreLabel> doc, PrintWriter out) {
    final String background = flags.backgroundSymbol;
    String prevTag = background;
    for (Iterator<CoreLabel> wordIter = doc.iterator(); wordIter.hasNext(); ) {
      CoreLabel wi = wordIter.next();
      String tag = wi.getString(AnswerAnnotation.class);
      if ( ! tag.equals(prevTag)) {
        if ( ! prevTag.equals(background) && ! tag.equals(background)) {
          out.print("</");
          out.print(prevTag);
          out.print('>');
          out.print(wi.getString(BeforeAnnotation.class));
          out.print('<');
          out.print(tag);
          out.print('>');
        } else if ( ! prevTag.equals(background)) {
          out.print("</");
          out.print(prevTag);
          out.print('>');
          out.print(wi.getString(BeforeAnnotation.class));
        } else if ( ! tag.equals(background)) {
          out.print(wi.getString(BeforeAnnotation.class));
         out.print('<');
          out.print(tag);
          out.print('>');
        }
      } else {
        out.print(wi.getString(BeforeAnnotation.class));
      }
      out.print(wi.getString(CurrentAnnotation.class));
      String afterWS = wi.getString(AfterAnnotation.class);

      if ( ! tag.equals(background) && ! wordIter.hasNext()) {
        out.print("</");
        out.print(tag);
        out.print('>');
        prevTag = background;
      } else {
        prevTag = tag;
      }
      out.print(afterWS);
    }
  }

  private void printAnswersTokenizedInlineXML(List<CoreLabel> doc, PrintWriter out) {
    final String background = flags.backgroundSymbol;
    String prevTag = background;
    boolean first = true;
    for (Iterator<CoreLabel> wordIter = doc.iterator(); wordIter.hasNext(); ) {
      CoreLabel wi = wordIter.next();
      String tag = wi.getString(AnswerAnnotation.class);
      if ( ! tag.equals(prevTag)) {
        if ( ! prevTag.equals(background) && ! tag.equals(background)) {
          out.print("</");
          out.print(prevTag);
          out.print("> <");
          out.print(tag);
          out.print('>');
        } else if ( ! prevTag.equals(background)) {
          out.print("</");
          out.print(prevTag);
          out.print("> ");
        } else if ( ! tag.equals(background)) {
          if ( ! first) {
            out.print(' ');
          }
          out.print('<');
          out.print(tag);
          out.print('>');
        }
      } else {
        if ( ! first) {
          out.print(' ');
        }
      }
      first = false;
      out.print(wi.getString(CurrentAnnotation.class));

      if ( ! wordIter.hasNext()) {
        if ( ! tag.equals(background)) {
          out.print("</");
          out.print(tag);
          out.print('>');
        }
        out.print(' ');
        prevTag = background;
      } else {
        prevTag = tag;
      }
    }
  }

}

/*
This is old stuff from a brief period when this DocumentReaderAndWriter tried
to handle treating SGML tags as part of white space, even though they were
returned as tokens by the tokenizer. If this is to be revived, it seems like
this handling should be moved down into the tokenizer.

These first two class declarations used to be in CoreAnnotations.  The rest
used to be in this class.

  public static class PrevSGMLAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class AfterSGMLAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }



  public Iterator<List<CoreLabel>> getIterator(Reader r) {
    PTBTokenizer<CoreLabel> ptb = PTBTokenizer.newPTBTokenizer(r, false, true);
    List firstSplit = new ArrayList();
    List<CoreLabel> d = new ArrayList<CoreLabel>();

    while (ptb.hasNext()) {
      CoreLabel w = ptb.next();
      Matcher m = sgml.matcher(w.word());
      if (m.matches()) {
        if (d.size() > 0) {
          firstSplit.add(d);
          d = new ArrayList<CoreLabel>();
        }
        firstSplit.add(w);
        continue;
      }
      d.add(w);
    }
    if (d.size() > 0) {
      firstSplit.add(d);
    }

    List secondSplit = new ArrayList();
    for (Object o : firstSplit) {
      if (o instanceof List) {
        secondSplit.addAll(wts.process((List) o));
      } else {
        secondSplit.add(o);
      }
    }

    String prevTags = "";
    CoreLabel lastWord = null;

    List<List<CoreLabel>> documents = new ArrayList<List<CoreLabel>>();

    boolean first = true;

    for (Object o : secondSplit) {
      if (o instanceof List) {
        List doc = (List) o;
        List<CoreLabel> document = new ArrayList<CoreLabel>();
        int pos = 0;
        for (Iterator wordIter = doc.iterator(); wordIter.hasNext(); pos++) {
          CoreLabel w = (CoreLabel) wordIter.next();
          w.set(PositionAnnotation.class, Integer.toString(pos));
          if (first && prevTags.length() > 0) {
            w.set(PrevSGMLAnnotation.class, prevTags);
          }
          first = false;
          lastWord = w;
          document.add(w);
        }
        documents.add(document);
      } else {
        //String tag = ((Word) o).word();
        CoreLabel word = (CoreLabel) o;
        String tag = word.before() + word.current();
        if (first) {
          System.err.println(word);
          prevTags = tag;
        } else {
          String t = lastWord.getString(AfterSGMLAnnotation.class);
          tag = t + tag;
          lastWord.set(AfterSGMLAnnotation.class, tag);
        }
      }
    }

    // this is a hack to deal with the incorrect assumption in the above code that
    // SGML only occurs between sentences and never inside of them.
    List<CoreLabel> allWords = new ArrayList<CoreLabel>();
    for (List<CoreLabel> doc : documents) {
      allWords.addAll(doc);
    }

    List<List<CoreLabel>> documentsFinal = wts.process(allWords);
    System.err.println(documentsFinal.get(0).get(0));
    System.exit(0);

    return documentsFinal.iterator();
//    return documents.iterator();
  }


  public void printAnswersInlineXML(List<CoreLabel> doc, PrintWriter out) {
    final String background = flags.backgroundSymbol;
    String prevTag = background;
    for (Iterator<CoreLabel> wordIter = doc.iterator(); wordIter.hasNext(); ) {
      CoreLabel wi = wordIter.next();
      String prev = wi.getString(PrevSGMLAnnotation.class);
      out.print(prev);
      if (prev.length() > 0) {
        prevTag = background;
      }
      String tag = wi.getString(AnswerAnnotation.class);
      if ( ! tag.equals(prevTag)) {
        if ( ! prevTag.equals(background) && ! tag.equals(background)) {
          out.print("</");
          out.print(prevTag);
          out.print('>');
          out.print(wi.getString(BeforeAnnotation.class));
          out.print('<');
          out.print(tag);
          out.print('>');
        } else if ( ! prevTag.equals(background)) {
          out.print("</");
          out.print(prevTag);
          out.print('>');
          out.print(wi.getString(BeforeAnnotation.class));
        } else if ( ! tag.equals(background)) {
          out.print(wi.getString(BeforeAnnotation.class));
         out.print('<');
          out.print(tag);
          out.print('>');
        }
      } else {
        out.print(wi.getString(BeforeAnnotation.class));
      }
      out.print(wi.getString(CurrentAnnotation.class));
      String after = wi.getString(AfterSGMLAnnotation.class);
      String afterWS = wi.getString(AfterAnnotation.class);

      if ( ! tag.equals(background) && ( ! wordIter.hasNext() || after.length() > 0)) {
        out.print("</");
        out.print(tag);
        out.print('>');
        prevTag = background;
      } else {
        prevTag = tag;
      }
      out.print(afterWS);
      out.print(after);
    }
  }





*/