package edu.stanford.nlp.ling;

import java.io.Serializable;


/**
 * Interface for Objects which can be described by their features.
 * These objects can also be Serialized (for insertion into a file database).
 *
 * @author Sepandar Kamvar (sdkamvar@stanford.edu)
 * 
 * @author Sarah Spikes (sdspikes@cs.stanford.edu) (Templatization)
 *
 * @param <L> The type of the labels in the Datum
 * @param <F> The type of the features in the Datum
 */
public abstract interface Datum<L, F> extends Serializable, Featurizable<F>, Labeled<L> {
}




