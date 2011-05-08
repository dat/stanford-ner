package edu.stanford.nlp.ling;

import java.util.List;
import java.util.Map;

import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

/**
 * <p>Set of common annotations for {@link CoreMap}'s.  The classes defined
 * here are typesafe keys for getting and setting annotation values.
 * These classes need not be instantiated outside of this class.  e.g
 * {@link WordAnnotation}.class serves as the key and a <code>String</code>
 * serves as the value containing the corresponding word.</p>
 *
 * <p>New types of {@link CoreAnnotation} can be defined anywhere that is
 * convenient in the source tree - they are just classes.  This file exists
 * to hold widely used "core" annotations and others inherited from the
 * {@link Label} family.  In general, most keys should be placed in this file
 * as they may often be reused throughout the code.  This architecture allows
 * for flexibility, but in many ways it should be considered as equivalent
 * to an enum in which everything should be defined</p>
 *
 * <p>The getType method required by CoreAnnotation must return the same class
 * type as its value type parameter.  It feels like one should be able to get
 * away without that method, but because Java erases the generic type signature,
 * that info disappears at runtime.  See {@link ValueAnnotation} for an
 * example.</p>
 *
 * @author dramage
 * @author rafferty
 */
public class CoreAnnotations {

  private CoreAnnotations() {} // only static members

  /**
   * These are the keys hashed on by IndexedWord
   */
  /**
   * This refers to the unique identifier for a "document", where document
   * may vary based on your application.
   */
  public static class DocIDAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * This is generally an individual word or feature index - it is local,
   * and may not be uniquely identifying without other identifiers such
   * as sentence and doc.  However, if these are the same, the index annotation
   * should be a unique identifier for differentiating objects.
   */
  public static class IndexAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {  return Integer.class; } }

  /**
   * Unique identifier within a document for a given sentence.
   */
  public static class SentenceIndexAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {  return Integer.class; } }

  /**
   * Contains the "value" - an ill-defined string used widely in MapLabel.
   */
  public static class ValueAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * String value of the word.
   */
  public static class WordAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }


  public static class CategoryAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * Annotation for the whitespace characters appearing before this word.
   * This can be filled in by the tokenizer so that the original text
   * string can be reconstructed.
   */
  public static class BeforeAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * Annotation for the whitespace characters appear after this word.
   * This can be filled in by the tokenizer so that the original text
   * string can be reconstructed.
   */
  public static class AfterAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * Penn POS tag in String form
   */
  public static class TagAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * CoNLL dep parsing - coarser POS tags.
   */
  public static class CoarseTagAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * CoNLL dep parsing - the dependency type
   */
  public static class CoNLLDepAnnotation implements CoreAnnotation<CoreMap> {
    public Class<CoreMap> getType() {  return CoreMap.class; } }

  /**
   * CoNLL SRL/dep parsing - whether the word is a predicate
   */
  public static class CoNLLPredicateAnnotation implements CoreAnnotation<Boolean> {
    public Class<Boolean> getType() {  return Boolean.class; } }

  /**
   * CoNLL SRL/dep parsing - map which, for the current word, specifies its
   * specific role for each predicate
   */
  public static class CoNLLSRLAnnotation implements CoreAnnotation<Map> {
    public Class<Map> getType() {  return Map.class; } }

  /**
   * CoNLL dep parsing - the dependency type
   */
  public static class CoNLLDepTypeAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * CoNLL dep parsing - the index of the word which is the parent of this word in the
   * dependency tree
   */
  public static class CoNLLDepParentIndexAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {  return Integer.class; } }

  /**
   * From the invertible PTB tokenizer - the actual, original current word.
   * That is, the tokenizer may normalize the token form to match what appears
   * in the PTB, but this key will hold the original characters.
   */
  public static class CurrentAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * character offset of the begining of the word.  PRESENTLY UNUSED.
   */
  public static class CharacterOffsetStart implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {  return Integer.class; } }

  /**
   * character offset of the end of the word.  PRESENTLY UNUSED.
   */
  public static class CharacterOffsetEnd implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {  return Integer.class; } }

  /**
   * Lemma for the word this label represents
   */
  public static class LemmaAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * Inverse document frequency of the word this label represents
   */
  public static class IDFAnnotation implements CoreAnnotation<Double> {
    public Class<Double> getType() {  return Double.class; } }


  /**
   * Keys from AbstractMapLabel (descriptions taken from that class)
   */
  /**
   * The standard key for storing a projected category in the map, as a String.
   * For any word (leaf node), the projected category is the syntactic category
   * of the maximal constituent headed by the word.  Used in SemanticGraph.
   */
  public static class ProjectedCategoryAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The standard key for a propbank label which is of type Argument
   */
  public static class ArgumentAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * Another key used for propbank - to signify core arg nodes or predicate nodes
   */
  public static class MarkingAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The standard key for Semantic Head Word which is a String
   */
  public static class SemanticHeadWordAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The standard key for Semantic Head Word POS which is a String
   */
  public static class SemanticHeadTagAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * Probank key for the Verb sense given in the Propbank Annotation, should
   * only be in the verbnode
   */
  public static class VerbSenseAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The standard key for storing category with functional tags.
   */
  public static class CategoryFunctionalTagAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * the standard key for the NER label. (e.g., DATE, PERSON, etc.)
   */
  public static class NERAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * POS tag for the NER (Note: not sure why this differs from "tag" annotation
   */
  public static class NERTagAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * This is an NER ID annotation (in case the all caps parsing didn't work out for you...)
   */
  public static class NERIDAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The key for the normalized value of numeric named entities.
   */
  public static class NormalizedNERAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public enum SRL_ID { ARG, NO, ALL_NO, REL }

  /**
   * The key for semantic role labels (Note: please add to this description if you use this key)
   */
  public static class SRLIDAnnotation implements CoreAnnotation<SRL_ID> {
    public Class<SRL_ID> getType() {  return SRL_ID.class; } }

/**
   * the standard key for the coref label.
   */
  public static class CorefAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /** The standard key for the "shape" of a word: a String representing
   *  the type of characters in a word, such as "Xx" for a capitalized word.
   *  See {@link edu.stanford.nlp.process.WordShapeClassifier} for functions
   *  for making shape strings.
   */
  public static class ShapeAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The Standard key for storing the left terminal number relative to the
   * root of the tree of the leftmost terminal dominated by the current node
   */
  public static class LeftTermAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {  return Integer.class; } }

  /**
   * The standard key for the parent which is a String
   */
  public static class ParentAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class INAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The standard key for span which is an IntPair
   */
  public static class SpanAnnotation implements CoreAnnotation<IntPair> {
    public Class<IntPair> getType() {  return IntPair.class; } }

  /**
   * The standard key for the answer which is a String
   */
  public static class AnswerAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The standard key for gold answer which is a String
   */
  public static class GoldAnswerAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The standard key for the features which is a Collection
   */
  public static class FeaturesAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The standard key for the semantic interpretation
   */
  public static class InterpretationAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The standard key for the semantic role label of a phrase.
   */
  public static class RoleAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The standard key for the gazetteer information
   */
  public static class GazetteerAnnotation implements CoreAnnotation<List<String>> {
    @SuppressWarnings({"unchecked", "RedundantCast"})
    public Class<List<String>> getType() {  return (Class) List.class; } }

  /**
   * Morphological stem of the word this label represents
   */
  public static class StemAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class PolarityAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }




  /**
   * for Chinese: character level information, segmentation
   */
  public static class ChineseCharAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class ChineseOrigSegAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class ChineseSegAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /** Not sure exactly what this is, but it is different from ChineseSegAnnotation and seems to indicate if the text is segmented */
  public static class ChineseIsSegmentedAnnotation implements CoreAnnotation<Boolean> {
    public Class<Boolean> getType() {  return Boolean.class; } }

  /** The character offset of first character of a token in source. */
  public static class BeginPositionAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {  return Integer.class; } }

  /** The character offset after the last character of a token in source. */
  public static class EndPositionAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {  return Integer.class; } }

  /**
   * Key for relative value of a word - used in RTE
   */
  public static class CostMagnificationAnnotation implements CoreAnnotation<Double> {
    public Class<Double> getType() {  return Double.class; } }

  public static class WordSenseAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }


  public static class SRLInstancesAnnotation implements CoreAnnotation<List<List<Pair<String,Pair>>>> {
    @SuppressWarnings({"unchecked", "RedundantCast"})
    public Class<List<List<Pair<String,Pair>>>> getType() {  return (Class) List.class; } }

  /**
   * Used by RTE to track number of text sentences, to determine when hyp sentences begin.
   */
  public static class NumTxtSentencesAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {  return Integer.class; } }

  /**
   * Used in Trees
   */
  public static class TagLabelAnnotation implements CoreAnnotation<Label> {
    public Class<Label> getType() {  return Label.class; } }

  /**
   * Used in CRFClassifier stuff
   * PositionAnnotation should possibly be an int - it's present as either an int or string depending on context
   * CharAnnotation may be "CharacterAnnotation" - not sure
   */
  public static class DomainAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class PositionAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class CharAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  //note: this is not a catchall "unknown" annotation but seems to have a specific meaning for sequence classifiers
  public static class UnknownAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class IDAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  //possibly this should be grouped with gazetteer annotation - original key was "gaz"
  public static class GazAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class PossibleAnswersAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class DistSimAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class AbbrAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class ChunkAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class GovernorAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class AbgeneAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class GeniaAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class AbstrAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class FreqAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class DictAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class WebAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class FemaleGazAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class MaleGazAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class LastGazAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /** it really seems like this should have a different name or else be a boolean */
  public static class IsURLAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class EntityTypeAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /** it really seems like this should have a different name or else be a boolean */
  public static class IsDateRangeAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class PredictedAnswerAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /** Seems like this could be consolidated with something else... */
  public static class OriginalAnswerAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /** Seems like this could be consolidated with something else... */
  public static class OriginalCharAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class UTypeAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {  return Integer.class; } }

  public static class EntityRuleAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class SectionAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class WordPosAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class ParaPosAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class SentencePosAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  //Why do both this and sentenceposannotation exist? I don't know, but one class
  //uses both so here they remain for now...
  public static class SentenceIDAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class EntityClassAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class AnswerObjectAnnotation implements CoreAnnotation<Object> {
    public Class<Object> getType() {  return Object.class; } }



  /**
   * Used in Task3 Pascal system
   */
  public static class BestCliquesAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class BestFullAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class LastTaggedAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * Used in wsd.supwsd package
   */
  public static class LabelAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class NeighborsAnnotation implements CoreAnnotation<List<Pair<WordLemmaTag,String>>> {
    @SuppressWarnings({"unchecked", "RedundantCast"})
    public Class<List<Pair<WordLemmaTag,String>>> getType() {  return (Class) List.class; } }

  public static class ContextsAnnotation implements CoreAnnotation<List<Pair<String, String>>> {
    @SuppressWarnings({"unchecked", "RedundantCast"})
    public Class<List<Pair<String, String>>> getType() {  return (Class) List.class; } }

  public static class DependentsAnnotation implements CoreAnnotation<List<Pair<Triple<String, String, String>, String>>> {
    @SuppressWarnings({"unchecked", "RedundantCast"})
    public Class<List<Pair<Triple<String, String, String>, String>>> getType() {  return (Class) List.class; } }

  public static class WordFormAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class TrueTagAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class SubcategorizationAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class BagOfWordsAnnotation implements CoreAnnotation<List<Pair<String,String>>> {
    @SuppressWarnings({"unchecked", "RedundantCast"})
    public Class<List<Pair<String, String>>> getType() {  return (Class) List.class; } }

  /**
   * Used in srl.unsup
   */
  public static class HeightAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class LengthAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }


  /**
   * Used in Gale2007ChineseSegmenter
   */
  public static class LBeginAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class LMiddleAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class LEndAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class D2_LBeginAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class D2_LMiddleAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class D2_LEndAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class UBlockAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /** Used in Chinese segmenters for whether there was space before a character. */
  public static class SpaceBeforeAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * Used in parser.discrim
   */

  /**
   * The base version of the parser state, like NP or VBZ or ...
   */
  public static class StateAnnotation implements CoreAnnotation<CoreLabel> {
    public Class<CoreLabel> getType() {  return CoreLabel.class; } }

  /**
   * used in binarized trees to say the name of the most recent child
   */
  public static class PrevChildAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * used in binarized trees to specify the first child in the rule for which
   * this node is the parent
   */
  public static class FirstChildAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * whether the node is the parent in a unary rule
   */
  public static class UnaryAnnotation implements CoreAnnotation<Boolean> {
    public Class<Boolean> getType() {  return Boolean.class; } }

  /**
   * annotation stolen from the lex parser
   */
  public static class DoAnnotation implements CoreAnnotation<Boolean> {
    public Class<Boolean> getType() {  return Boolean.class; } }

  /**
   * annotation stolen from the lex parser
   */
  public static class HaveAnnotation implements CoreAnnotation<Boolean> {
    public Class<Boolean> getType() {  return Boolean.class; } }

  /**
   * annotation stolen from the lex parser
   */
  public static class BeAnnotation implements CoreAnnotation<Boolean> {
    public Class<Boolean> getType() {  return Boolean.class; } }

  /**
   * annotation stolen from the lex parser
   */
  public static class NotAnnotation implements CoreAnnotation<Boolean> {
    public Class<Boolean> getType() {  return Boolean.class; } }

  /**
   * annotation stolen from the lex parser
   */
  public static class PercentAnnotation implements CoreAnnotation<Boolean> {
    public Class<Boolean> getType() {  return Boolean.class; } }

  /**
   * specifies the base state of the parent of this node in the parse tree
   */
  public static class GrandparentAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * The key for storing a Head word as a string rather than a pointer
   * (as in TreeCoreAnnotations.HeadWordAnnotation)
   */
  public static class HeadWordStringAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * Used in nlp.coref
   */
  public static class MonthAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class DayAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class YearAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  /**
   * Used in propbank.srl
   */
  public static class PriorAnnotation implements CoreAnnotation<Map<String, Double>> {
    @SuppressWarnings({"unchecked", "RedundantCast"})
    public Class<Map<String, Double>> getType() {  return (Class) Map.class; } }

  public static class SemanticWordAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class SemanticTagAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {  return String.class; } }

  public static class CovertIDAnnotation implements CoreAnnotation<List<IntPair>> {
    @SuppressWarnings({"unchecked", "RedundantCast"})
    public Class<List<IntPair>> getType() {  return (Class) List.class; } }


  public static class ArgDescendentAnnotation implements CoreAnnotation<Pair<String, Double>> {
    @SuppressWarnings({"unchecked", "RedundantCast"})
    public Class<Pair<String, Double>> getType() {  return (Class) Pair.class; } }

  /**
   * Used in nlp.trees
   */
  public static class CopyAnnotation implements CoreAnnotation<Boolean> {
    public Class<Boolean> getType() {  return Boolean.class; } }

  public static class ValueLabelAnnotation implements CoreAnnotation<Label> {
    public Class<Label> getType() {  return Label.class; } }

}


