package edu.berkeley.nlp.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Open address hash map with linear probing. Assumes keys are non-negative
 * (uses -1 internally for empty key). Returns 0.0 for keys not in the map.
 * 
 * Much more memory- and speed-efficient than Counter<Integer>.
 * 
 * @author adampauls
 * 
 */
public class IntCounter {

  private int[] keys;

  private double[] values;

  private int size = 0;

  private static final int EMPTY_KEY = -1;

  private double maxLoadFactor = 0.5;

  private boolean sorted = false;

  public IntCounter() {
    this(5);
  }

  public double[] toArray(int length) {
    double[] result = new double[length];
    for (Map.Entry<Integer, Double> entry : entries()) {
      result[entry.getKey()] = entry.getValue();
    }
    return result;
  }

  public void setLoadFactor(double loadFactor) {
    this.maxLoadFactor = loadFactor;
    ensureCapacity(values.length);
  }

  public IntCounter(int initCapacity_) {
    int initCapacity = toSize(initCapacity_);
    keys = new int[initCapacity];
    values = new double[initCapacity];
    Arrays.fill(keys, EMPTY_KEY);
  }

  public static IntCounter wrapArray(double[] arrayToWrap, int size) {
    return new IntCounter(arrayToWrap, size);
  }

  private IntCounter(double[] arrayToWrap, int size) {
    this.values = arrayToWrap;
    this.keys = null;
    this.size = size;
  }

  public void toSorted() {
    sorted = true;
    int[] newKeys = new int[size];
    double[] newValues = new double[size];
    List<Entry> sortedEntries = new ArrayList<Entry>(size);
    for (java.util.Map.Entry<Integer, Double> e : entries()) {
      sortedEntries.add((Entry) e);
    }
    Collections.sort(sortedEntries, new Comparator<Entry>() {

      public int compare(Entry o1, Entry o2) {
        return Double.compare(o1.key, o2.key);
      }
    });
    int k = 0;
    for (Entry e : sortedEntries) {
      newKeys[k] = e.getKey();
      newValues[k] = e.getValue();
      k++;
    }
    keys = newKeys;
    values = newValues;
  }

  /**
   * @param initCapacity_
   * @return
   */
  private int toSize(int initCapacity_) {
    return Math.max(5, (int) (initCapacity_ / maxLoadFactor) + 1);
  }

  public IntCounter(Counter<Integer> c) {
    final double d = c.size() / maxLoadFactor;
    final int initSize = Math.max(5, (int) d + 1);
    keys = new int[initSize];
    values = new double[keys.length];
    Arrays.fill(keys, EMPTY_KEY);
    for (Map.Entry<Integer, Double> entry : c.getEntrySet()) {
      put(entry.getKey(), entry.getValue());
    }
  }

  public boolean put(int k, double v) {
    checkNotImmutable();
    assert !Double.isNaN(v);
    if (size / (double) keys.length > maxLoadFactor) {
      rehash();
    }
    return putHelp(k, v, keys, values);

  }

  /**
	 * 
	 */
  private void checkNotImmutable() {
    if (keys == null)
      throw new RuntimeException("Cannot change wrapped IntCounter");
    if (sorted)
      throw new RuntimeException("Cannot change sorted IntCounter");
  }

  /**
	 * 
	 */
  private void rehash() {
    final int length = keys.length * 2 + 1;
    rehash(length);
  }

  /**
   * @param length
   */
  private void rehash(final int length) {
    checkNotImmutable();
    int[] newKeys = new int[length];
    double[] newValues = new double[length];
    Arrays.fill(newKeys, EMPTY_KEY);
    size = 0;
    for (int i = 0; i < keys.length; ++i) {
      int curr = keys[i];
      if (curr != EMPTY_KEY) {
        double val = values[i];
        putHelp(curr, val, newKeys, newValues);
      }
    }
    keys = newKeys;
    values = newValues;
  }

  /**
   * @param k
   * @param v
   */
  private boolean putHelp(int k, double v, int[] keyArray, double[] valueArray) {
    checkNotImmutable();
    assert k >= 0;
    int pos = getInitialPos(k, keyArray);
    int currKey = keyArray[pos];
    while (currKey != EMPTY_KEY && currKey != k) {
      pos++;
      if (pos == keyArray.length)
        pos = 0;
      currKey = keyArray[pos];
    }

    valueArray[pos] = v;
    if (currKey == EMPTY_KEY) {
      size++;
      keyArray[pos] = k;
      return true;
    }
    return false;
  }

  /**
   * @param k
   * @param keyArray
   * @return
   */
  private int getInitialPos(final int k, final int[] keyArray) {
    if (keyArray == null)
      return k;
    int hash = k;
    if (hash < 0)
      hash = -hash;
    int pos = hash % keyArray.length;
    return pos;
  }

  public double get(int k) {
    int pos = find(k);
    if (pos == EMPTY_KEY)
      return 0.0;

    return values[pos];
  }

  /**
   * @param k
   * @return
   */
  private int find(int k) {
    if (keys == null) {
      return k < values.length ? k : EMPTY_KEY;
    } else if (sorted) {
      final int pos = Arrays.binarySearch(keys, k);
      return pos < 0 ? EMPTY_KEY : pos;

    } else {
      final int[] localKeys = keys;
      final int length = localKeys.length;
      int pos = getInitialPos(k, localKeys);
      int curr = localKeys[pos];
      while (curr != EMPTY_KEY && curr != k) {
        pos++;
        if (pos == length)
          pos = 0;
        curr = localKeys[pos];
      }
      return pos;
    }
  }

  public double dotProduct(Counter<Integer> other) {
    double sum = 0.0;

    final int[] localKeys = keys;
    final double[] localValues = values;
    for (int i = 0; i < localValues.length; ++i) {
      int key = localKeys == null ? i : localKeys[i];
      if (key == EMPTY_KEY)
        continue;
      double val = localValues[i];
      if (val == 0.0)
        continue;
      final double d = other.getCount(key);
      sum += val * d;
    }
    return sum;

  }

  /**
   * @param sum
   * @return
   */
  public double normSquared() {
    double sum = 0.0;
    final int[] localKeys = keys;
    final double[] localValues = values;
    for (int i = 0; i < localValues.length; ++i) {
      int key = localKeys == null ? i : localKeys[i];
      if (key == EMPTY_KEY)
        continue;
      final double d = localValues[i];
      sum += d * d;
    }
    return sum;
  }

  public double dotProduct(IntCounter other) {
    double sum = 0.0;
    if (other == this) {
      sum = normSquared();
    } else {
      final int[] localKeys = keys;
      final double[] localValues = values;
      for (int i = 0; i < localValues.length; ++i) {
        int key = localKeys == null ? i : localKeys[i];
        if (key == EMPTY_KEY)
          continue;
        double val = localValues[i];
        if (val == 0.0)
          continue;
        final double d = other.getCount(key);
        sum += val * d;
      }
    }
    return sum;

  }

  public double dotProduct(double[] weights) {
    final int[] localKeys = keys;
    final double[] localValues = values;
    double sum = 0.0;
    for (int i = 0; i < localValues.length; ++i) {
      int key = localKeys == null ? i : localKeys[i];
      if (key == EMPTY_KEY)
        continue;
      double val = localValues[i];
      if (val == 0.0)
        continue;
      final double d = key >= weights.length ? 0.0 : weights[key];
      sum += val * d;
    }
    return sum;
  }

  public static class Entry implements Map.Entry<Integer, Double> {
    public Entry(int key, double value) {
      super();
      this.key = key;
      this.value = value;
    }

    public int key;

    public double value;

    public Integer getKey() {
      return key;
    }

    public Double getValue() {
      return value;
    }

    public Double setValue(Double value) {
      this.value = value;
      return this.value;
    }
  }

  private class EntryIterator extends MapIterator<Map.Entry<Integer, Double>> {
    public Entry next() {
      final int nextIndex = nextIndex();
      return new Entry(keys == null ? nextIndex : keys[nextIndex], values[nextIndex]);
    }
  }

  private class KeyIterator extends MapIterator<Integer> {
    public Integer next() {
      final int nextIndex = nextIndex();
      return keys == null ? nextIndex : keys[nextIndex];
    }
  }

  private class PrimitiveEntryIterator extends MapIterator<Entry> {
    public Entry next() {
      final int nextIndex = nextIndex();
      return new Entry(keys == null ? nextIndex : keys[nextIndex], values[nextIndex]);
    }
  }

  private abstract class MapIterator<E> implements Iterator<E> {
    public MapIterator() {
      end = keys == null ? size : values.length;
      next = -1;
      nextIndex();
    }

    public boolean hasNext() {
      return end > 0 && next < end;
    }

    int nextIndex() {
      int curr = next;
      do {
        next++;
      } while (next < end && keys != null && keys[next] == EMPTY_KEY);
      return curr;
    }

    public void remove() {
      throw new UnsupportedOperationException();
    }

    private int next, end;
  }

  public Iterable<Map.Entry<Integer, Double>> entries() {
    return CollectionUtils.iterable(new EntryIterator());
  }

  public double incrementCount(Integer k, double d) {
    checkNotImmutable();
    if (d == 0.0)
      return getCount(k);
    assert !Double.isNaN(d);
    int pos = find(k);
    if (keys[pos] == EMPTY_KEY)
      put(k, d);
    else
      values[pos] += d;
    return getCount(k);
  }

  public void incrementCount(int k, double d) {
    checkNotImmutable();
    if (d == 0.0)
      return;
    assert !Double.isNaN(d);
    int pos = find(k);
    if (keys[pos] == EMPTY_KEY)
      put(k, d);
    else
      values[pos] += d;

  }

  public <T extends Integer> void incrementAll(Counter<T> c, double d) {
    checkNotImmutable();
    assert !Double.isNaN(d);
    for (Map.Entry<T, Double> entry : c.getEntrySet()) {
      final double d2 = d * entry.getValue();
      if (d2 == 0.0)
        continue;
      incrementCount(entry.getKey(), d2);
    }
  }

  public void incrementAll(IntCounter c, double d) {
    checkNotImmutable();
    if (d == 0.0)
      return;
    for (int i = 0; i < c.values.length; ++i) {
      int key = c.keys == null ? i : c.keys[i];
      if (key == EMPTY_KEY)
        continue;
      final double v = d * c.values[i];
      incrementCount(key, v);
    }

  }

  public void incrementAll(IntCounter c) {
    incrementAll(c, 1.0);

  }

  public void incrementAll(int[] arr, double d) {
    checkNotImmutable();
    if (d == 0.0)
      return;
    for (int i = 0; i < arr.length; ++i) {
      int key = arr[i];
      if (key == EMPTY_KEY)
        continue;
      incrementCount(key, d);
    }
  }

  public void incrementAll(int[] arr) {
    incrementAll(arr, 1.0);
  }

  public <T extends Integer> void incrementAll(Counter<T> c) {
    // ensureCapacity(c.size() + 1);
    incrementAll(c, 1.0);
  }

  public void ensureCapacity(int capacity) {
    checkNotImmutable();
    int newSize = toSize(capacity);
    if (newSize > keys.length) {
      rehash(newSize);
    }
  }

  public Counter<Integer> toCounter() {
    Counter<Integer> c = new Counter<Integer>();
    for (Map.Entry<Integer, Double> entry : entries()) {
      c.setCount(entry.getKey(), entry.getValue());
    }
    return c;
  }

  public void scale(double d) {
    for (int i = 0; i < keys.length; ++i) {
      values[i] *= d;
    }
  }

  public double getCount(Integer k) {
    return get(k);
  }

  public double getCount(int k) {
    return get(k);
  }

  public void setCount(Integer k, double d) {
    put(k, d);
  }

  public void setCount(int k, double d) {
    put(k, d);
  }

  public int size() {
    return size;
  }

  public double totalCount() {
    double totalCount = 0.0;
    for (Map.Entry<Integer, Double> entry : entries()) {
      totalCount += entry.getValue();
    }
    return totalCount;
  }

  public Iterable<Entry> primitiveEntries() {
    return CollectionUtils.iterable(new PrimitiveEntryIterator());
  }

  /**
   * Increments the weights, not c
   * 
   * @param weights
   */
  public static void incrementDenseArray(double[] weights, final IntCounter c, final double d) {
    if (d == 0.0)
      return;
    final int[] localKeys = c.keys;
    final double[] localValues = c.values;
    for (int i = 0; i < localValues.length; ++i) {
      final double val = localValues[i];
      if (val == 0.0)
        continue;
      final int key = localKeys == null ? i : localKeys[i];
      if (key == EMPTY_KEY)
        continue;
      final double v = d * val;
      if (Double.isNaN(v) || Double.isInfinite(v)) {
        @SuppressWarnings("unused")
        int x = 5;
      }
      weights[key] += v;
    }
  }

  public Iterable<Integer> keySet() {
    return CollectionUtils.iterable(new KeyIterator());
  }

  public void clear() {
    Arrays.fill(keys, EMPTY_KEY);
    Arrays.fill(values, 0.0);
    size = 0;

  }

}
