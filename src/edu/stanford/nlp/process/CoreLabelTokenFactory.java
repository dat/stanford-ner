package edu.stanford.nlp.process;


import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.BeginPositionAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.EndPositionAnnotation;

/**
 * Constructs {@link CoreLabel}s as Strings with a corresponding BEGIN and END position.
 *
 * @author Anna Rafferty
 */
public class CoreLabelTokenFactory implements LexedTokenFactory<CoreLabel> {
  boolean addIndices = true;
  
  /**
   * Constructor for a new token factory which will add in the word, the "current" annotation, and the begin/end position annotations.
   */
  public CoreLabelTokenFactory() {
    super();
  }
  
  /**
   * Constructor that allows one to choose if index annotation indicating begin/end position will be included in
   * the label
   * @param addIndices if true, begin and end position annotations will be included (this is the default)
   */
  public CoreLabelTokenFactory(boolean addIndices) {
    super();
    this.addIndices = addIndices;
  }

  /**
   * Constructs a CoreLabel as a String with a corresponding BEGIN and END position.
   * (Does not take substr).
   */
  public CoreLabel makeToken(String str, int begin, int length) {
    CoreLabel cl = new CoreLabel();
    cl.setWord(str);
    cl.setCurrent(str);
    if(addIndices) {
      cl.set(BeginPositionAnnotation.class, begin);
      cl.set(EndPositionAnnotation.class, begin+length);
    }
    return cl;
  }

}
