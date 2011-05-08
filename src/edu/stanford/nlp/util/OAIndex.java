package edu.stanford.nlp.util;

import java.io.*; import java.util.*;

/**
 * Open addressing backed index for arbitrary object types.
 * Includes support for both traditional matching of keys values based on
 * the equals() method as well as 'identity' matching where by keys are 
 * only compared with '=='. 
 * 
 * WARNING: This is currently experimental code. It exists since in 
 * theory open addressing hashing should be more efficient 
 * for hashing int values than bucket-chain hashing, which is used
 * to back java.util.HashMap. 
 * 
 * To do list:
 *   - in fear of user stupidity, i think HashMap re-hashes the hash 
 *     values returned by objects, should we? 
 *   - finalize interface 
 *   - rigorous benchmarks
 *   - unit tests
 *
 * @author <a href="mailto:daniel.cer@cs.colorado.edu">Daniel Cer</a> 
 */
public class OAIndex<K> implements IndexInterface<K> {
    
  private static final long serialVersionUID = 127L;
    
  static final int INIT_SZ = 1<<10;
  static final double MAX_LOAD = 0.60;

  // Index<K> sanityIndex = new Index<K>();
  
  private final boolean identityHash; 
  private Object[] keys; private int[] values; private int mask;
  private int[] hashCodes;
  private int[] reverseIndex;
  int maxIndex; int load;

  public OAIndex() { identityHash = false; init(); } 
  public OAIndex(boolean identityHash) { this.identityHash = identityHash;
    init(); }
 
  @SuppressWarnings("unchecked")
  public Set<K> keySet() {
	  Set<K> keySet = new HashSet<K>();
	  for (int i = 0; i < keys.length; i++) {
		  if (keys[i] == null) continue;
		  keySet.add((K) keys[i]);
	  }
	  return keySet;
  }
  
  public int maxIndex() {
	  return maxIndex;
  }
  
  public int boundOnMaxIndex() {
	  return keys.length;
  }
  private void init() {
    keys = new Object[INIT_SZ]; // since we can't create an array of type K[];
    values = new int[INIT_SZ];
    hashCodes = new int[INIT_SZ];
    reverseIndex = new int[INIT_SZ]; Arrays.fill(reverseIndex, -1);
    mask = INIT_SZ - 1;
  }

  private int supplementalHash(int h) {
      // use the same supplemental hash function used by HashMap
      return ((h << 7) - h + (h >>> 9) + (h >>> 17));
  }
  
  private int findPos(Object e, boolean add) { 
    int hashCode = supplementalHash(e.hashCode());    
    int idealIdx = hashCode & mask;
    
    if (identityHash) { 
      for (int i = 0, idx = idealIdx; i < keys.length; i++, idx++) {
        if (idx >= keys.length) idx = 0;
        if (keys[idx] == null) return -idx-1;
        if (keys[idx] == e) return idx;        
      }
    } else {	
	for (int i = 0, idx = idealIdx; i < keys.length; i++, idx++) {
	    if (idx >= keys.length) idx = 0;
	    if (keys[idx] == null) return -idx-1;
	    if (hashCodes[idx] != hashCode) continue;    
	    if (keys[idx].equals(e)) return idx;
        }        
    }
    return -keys.length-1;
  }
  
  @SuppressWarnings("unchecked")
  public K get(int idx) {
      int pos = reverseIndex[idx];
      if (pos == -1) return null;
      return (K)keys[pos];
  }
  
  private void sizeUp() {      
    int newSize = keys.length<<1;
    mask = newSize-1;
    //System.err.printf("size up to: %d\n", newSize);
    Object[] oldKeys = keys; int[] oldValues = values; int[] oldHashCodes = hashCodes;
    keys = new Object[newSize]; values = new int[newSize];
    reverseIndex = new int[newSize]; Arrays.fill(reverseIndex, -1);
    hashCodes = new int[newSize];
    for (int i = 0; i < oldKeys.length; i++) { if (oldKeys[i]==null) continue;
      int pos = -findPos(oldKeys[i], true)-1; 
      keys[pos] = oldKeys[i]; values[pos] = oldValues[i];
      reverseIndex[values[pos]] = pos;
      hashCodes[pos] = oldHashCodes[i];      
    }
  } 
  
@SuppressWarnings("unused")
private int getSearchOffset(int pos, Object key) {
      int idealIdx = supplementalHash(key.hashCode()) & mask;      
      int distance;
      if (idealIdx < pos) {
  	distance = pos + keys.length - idealIdx;
      } else {
  	distance = pos - idealIdx;
      }
      return distance;
  }
  
  private int add(K key, int pos) {
    if ((load++)/(double)keys.length > MAX_LOAD) { 
      sizeUp();
      pos = -findPos(key, true)-1;
    }
    keys[pos] = key; values[pos] = maxIndex++;
    reverseIndex[values[pos]] = pos;
    hashCodes[pos] = supplementalHash(key.hashCode());
    return maxIndex-1;
  }
  
  @Override
  @SuppressWarnings("unchecked")
  public String toString() { 
    Set keySet = new TreeSet();
    for (int i = 0; i < keys.length; i++) { if (keys[i] == null) continue;
      keySet.add(keys[i]); }
    StringBuffer sb = new StringBuffer();
    sb.append("[");
    for (Object k : keySet) {
     sb.append(k).append(":").append(indexOf((K)k)).append(" ");
    } sb.append("]");
    return sb.toString();
  } 

  public int indexOf(K key) { 
      int pos = findPos(key, false);
      if (pos < 0) return -1;
      return values[pos]; 
  }
  
  public boolean contains(Object key) {
      int pos = findPos(key, false);
      if (pos < 0) return false;
      return true;
  }

  public int indexOf(K key, boolean add) {
    int pos = findPos(key, add);
    if (pos >= 0) return values[pos];
    if (!add) return -1;
    //System.out.printf("adding: %s %d\n", key, -pos-1);
    return add(key, -pos-1); /*
    if (pos != sanityIndex.indexOf(key, true)) {
	System.err.printf("%d != %d", pos, sanityIndex.indexOf(key));
	System.exit(-1);
    } */    
  } 

  static public void main(String[] args) throws IOException { 
    if (args.length != 1) {
      System.err.printf("Usage:\n\tjava ...OAIndex (text file to index)\n");
      System.exit(-1); }

    BufferedReader breader = new BufferedReader(new FileReader(args[0]));     
    OAIndex<String> oaindex = new OAIndex<String>();
    System.out.printf("Inserting tokens:\n");
    for (String line; (line = breader.readLine()) != null; ) { 
      String[] tokens = line.split("\\s");
      for (String token : tokens) { 
        oaindex.indexOf(token, true);  
        System.out.printf("%s: %d (get: %s)\n", token, oaindex.indexOf(token), 
          oaindex.get(oaindex.indexOf(token)));
      }  
    }
    System.out.println();
    System.out.printf("Final Index:\n%s\n", oaindex);
  }
  public int size() {
    return load;
  }
}
