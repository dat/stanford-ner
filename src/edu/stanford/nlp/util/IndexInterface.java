package edu.stanford.nlp.util;

import java.io.Serializable;

/**
 * Minimalist interface for implementations of Index.
 * 
 * This interface should allow Index and OAIndex to be used interchangeably 
 * in certain contexts.
 * 
 * Originally extracted from util.Index on 3/13/2007
 * 
 * @author Daniel Cer
 *
 * @param <E>
 */
public interface IndexInterface<E> extends Serializable {
    static final int INVALID_ENTRY = -1;
    
    /**
     * Returns the number of indexed objects.
     * @return the number of indexed objects.
     */
    public abstract int size();

    /**
     * Gets the object whose index is the integer argument.
     * @param i the integer index to be queried for the corresponding argument
     * @return the object whose index is the integer argument.
     */
    public abstract E get(int i);

    /**
     * Returns the integer index of the Object in the Index or -1 if the Object is not already in the Index.
     * @param o the Object whose index is desired.
     * @return the index of the Object argument.  Returns -1 if the object is not in the index.
     */
    public abstract int indexOf(E o);

    /**
     * Takes an Object and returns the integer index of the Object,
     * perhaps adding it to the index first.
     * Returns -1 if the Object is not in the Index.
     * (Note: indexOf(x, true) is the direct replacement for the number(x)
     * method in the old Numberer class.)
     * 
     * @param o the Object whose index is desired.
     * @param add Whether it is okay to add new items to the index
     * @return the index of the Object argument.  Returns -1 if the object is not in the index.
     */
    public abstract int indexOf(E o, boolean add);
    
    public boolean contains(Object o);

}