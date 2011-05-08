package edu.stanford.nlp.sequences;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.AnswerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.GoldAnswerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PositionAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.ShapeAnnotation;
import edu.stanford.nlp.objectbank.ObjectBank;
import edu.stanford.nlp.process.Americanize;
import edu.stanford.nlp.process.WordShapeClassifier;
import edu.stanford.nlp.util.AbstractIterator;
import java.util.*;
import java.util.regex.Pattern;


/**
 * This class is used to wrap the ObjectBank used by the sequence
 * models and is where any sort of general processing, like the IOB mapping
 * stuff and wordshape stuff, should go.
 * It checks the SeqClassifierFlags to decide what to do.
 * <p>
 * TODO: We should rearchitect this so that the FeatureFactory-specific
 * stuff is done by a callback to the relevant FeatureFactory.
 *
 * @author Jenny Finkel
 */
public class ObjectBankWrapper extends ObjectBank<List<CoreLabel>> {

  private static final long serialVersionUID = -3838331732026362075L;

  private SeqClassifierFlags flags;
  private ObjectBank<List<CoreLabel>> wrapped;
  private Set<String> knownLCWords;


  public ObjectBankWrapper(SeqClassifierFlags flags, ObjectBank<List<CoreLabel>> wrapped, Set<String> knownLCWords) {
    super(null,null);
    this.flags = flags;
    this.wrapped = wrapped;
    this.knownLCWords = knownLCWords;
  }


  @Override
  public Iterator<List<CoreLabel>> iterator() {
    Iterator<List<CoreLabel>> iter = new WrappedIterator(wrapped.iterator());

    // If using WordShapeClassifier, we have to make an extra pass through the
    // data before we really process it, so that we can build up the
    // database of known lower case words in the data.  We do that here.
    if ((flags.wordShape > WordShapeClassifier.NOWORDSHAPE) && (!flags.useShapeStrings)) {
      while (iter.hasNext()) {
        List<CoreLabel> doc = iter.next();
        for (CoreLabel fl : doc) {
          String word = fl.word();
          if (word.length() > 0) {
            char ch = word.charAt(0);
            if (Character.isLowerCase(ch)) {
              knownLCWords.add(word);
            }
          }
        }
      }
      iter = new WrappedIterator(wrapped.iterator());
    }
    return iter;
  }

  private class WrappedIterator extends AbstractIterator<List<CoreLabel>> {
    Iterator<List<CoreLabel>> wrappedIter;
    Iterator<List<CoreLabel>> spilloverIter;

    public WrappedIterator(Iterator<List<CoreLabel>> wrappedIter) {
      this.wrappedIter = wrappedIter;
    }

    @Override
    public boolean hasNext() {
      return wrappedIter.hasNext() ||
        (spilloverIter != null && spilloverIter.hasNext());
    }

    @Override
    public List<CoreLabel> next() {
      while (spilloverIter == null || !spilloverIter.hasNext()) {
        List<CoreLabel> doc = wrappedIter.next();
        List<List<CoreLabel>> docs = new ArrayList<List<CoreLabel>>();
        docs.add(doc);
        fixDocLengths(docs);
        spilloverIter = docs.iterator();
      }

      return processDocument(spilloverIter.next());
    }
  }

  public List<CoreLabel> processDocument(List<CoreLabel> doc) {
    if (flags.mergeTags) { mergeTags(doc); }
    if (flags.iobTags) { iobTags(doc); }
    doBasicStuff(doc);

    return doc;
  }

  private String intern(String s) {
    if (flags.intern) {
      return s.intern();
    } else {
      return s;
    }
  }


  private final Pattern monthDayPattern = Pattern.compile("Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|January|February|March|April|May|June|July|August|September|October|November|December", Pattern.CASE_INSENSITIVE);

  private String fix(String word) {
    if (flags.normalizeTerms || flags.normalizeTimex) {
      // Same case for days/months: map to lowercase
      if (monthDayPattern.matcher(word).matches()) {
        return word.toLowerCase();
      }
    }
    if (flags.normalizeTerms) {
      return Americanize.americanize(word, false);
    }
    return word;
  }


  private void doBasicStuff(List<CoreLabel> doc) {
    int position = 0;
    for (CoreLabel fl : doc) {

      // position in document
      fl.set(PositionAnnotation.class, Integer.toString((position++)));

      // word shape
      if ((flags.wordShape > WordShapeClassifier.NOWORDSHAPE) && (!flags.useShapeStrings)) {
        String s = intern(WordShapeClassifier.wordShape(fl.word(), flags.wordShape, knownLCWords));
        fl.set(ShapeAnnotation.class, s);
      }

      // normalizing and interning
      // was the following; should presumably now be
      // if ("CTBSegDocumentReader".equalsIgnoreCase(flags.documentReader)) {
      if ("edu.stanford.nlp.wordseg.Sighan2005DocumentReaderAndWriter".equalsIgnoreCase(flags.readerAndWriter)) {
        // for Chinese segmentation, "word" is no use and ignore goldAnswer for memory efficiency.
        fl.set(CharAnnotation.class,intern(fix(fl.get(CharAnnotation.class))));
      } else {
        fl.setWord(intern(fix(fl.word())));
        fl.set(GoldAnswerAnnotation.class, fl.get(AnswerAnnotation.class));
      }
    }
  }

  /**
   * Take a {@link List} of documents (which are themselves {@link List}s
   * of {@link CoreLabel}s) and if any are longer than the length
   * specified by flags.maxDocSize split them up.  It tries to be smart
   * and split on sentence bounaries, hard-coded to the English-specific token
   * '.'.
   * TODO: This implementation is broken. It fails on a zero length document:
   * after while loop it doesn't get added, as empty, it gets removed from the list,
   * and then i doesn't increment because no new document.
   */
  private void fixDocLengths(List<List<CoreLabel>> docs) {
    int maxSize = flags.maxDocSize;
    if (maxSize <= 0) {
      return;
    }

    for (int i = 0; i < docs.size(); i++) {
      List<CoreLabel> document = docs.get(i);
      List<List<CoreLabel>> newDocuments = new ArrayList<List<CoreLabel>>();
      while (document.size() > maxSize) {
        int splitIndex = 0;
        for (int j = maxSize; j > maxSize / 2; j--) {
          CoreLabel wi = document.get(j);
          if (wi.word().equals(".")) {
            splitIndex = j + 1;
            break;
          }
        }
        if (splitIndex == 0) {
          splitIndex = maxSize;
        }
        List<CoreLabel> newDoc = document.subList(0, splitIndex);
        newDocuments.add(newDoc);
        document = document.subList(splitIndex, document.size());
      }
      if ( ! document.isEmpty()) {
        newDocuments.add(document);
      }
      docs.remove(i);
      Collections.reverse(newDocuments);
      for (List<CoreLabel> item : newDocuments) {
        docs.add(i, item);
      }
      i += newDocuments.size() - 1;
    }
  }


  private void iobTags(List<CoreLabel> doc) {
    String lastTag = "";
    for (CoreLabel wi : doc) {
      String answer = wi.get(AnswerAnnotation.class);
      if (!answer.equals(flags.backgroundSymbol)) {
        int index = answer.indexOf('-');
        String prefix;
        String label;
        if (index < 0) {
          prefix = "";
          label = answer;
        } else {
          prefix = answer.substring(0,1);
          label = answer.substring(2);
        }

        if (!prefix.equals("B")) {
          if (!lastTag.equals(label)) {
            wi.set(AnswerAnnotation.class, "B-" + label);
          } else {
            wi.set(AnswerAnnotation.class, "I-" + label);
          }
        }
        lastTag = label;
      } else {
        lastTag = answer;
      }
    }
  }


  private void mergeTags(List<CoreLabel> doc) {
    for (CoreLabel wi : doc) {
      String answer = wi.get(AnswerAnnotation.class);
      if (!answer.equals(flags.backgroundSymbol) && answer.indexOf('-') >= 0) {
        answer = answer.substring(2);
      }
      wi.set(AnswerAnnotation.class, answer);
    }
  }


  // all the other the crap from ObjectBank
  @Override
  public boolean add(List<CoreLabel> o) { return wrapped.add(o); }
  @Override
  public boolean addAll(Collection<? extends List<CoreLabel>> c) { return wrapped.addAll(c); }
  @Override
  public void clear() { wrapped.clear(); }
  @Override
  public void clearMemory() { wrapped.clearMemory(); }
  public boolean contains(List<CoreLabel> o) { return wrapped.contains(o); }
  @Override
  public boolean containsAll(Collection<?> c) { return wrapped.containsAll(c); }
  @Override
  public boolean isEmpty() { return wrapped.isEmpty(); }
  @Override
  public void keepInMemory(boolean keep) { wrapped.keepInMemory(keep); }
  public boolean remove(List<CoreLabel> o) { return wrapped.remove(o); }
  @Override
  public boolean removeAll(Collection<?> c) { return wrapped.removeAll(c); }
  @Override
  public boolean retainAll(Collection<?> c) { return wrapped.retainAll(c); }
  @Override
  public int size() { return wrapped.size(); }
  @Override
  public Object[] toArray() { return wrapped.toArray(); }
  public List<CoreLabel>[] toArray(List<CoreLabel>[] o) { return wrapped.toArray(o); }

} // end class ObjectBankWrapper
