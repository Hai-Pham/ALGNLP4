package edu.berkeley.nlp.util;

import java.util.Arrays;
import java.util.Iterator;

/**
 * Open address hash map with linear probing. Maps Strings to int's. Note that
 * int's are assumed to be non-negative, and -1 is returned when a key is not
 * present.
 * 
 * @author adampauls
 * 
 */
public class TIntOpenHashMap<T>
{

	private T[] keys;

	private int[] values;

	private int size = 0;

	private final T EMPTY_KEY = null;

	private final double MAX_LOAD_FACTOR;

	public boolean put(T k, int v) {
		if (size / (double) keys.length > MAX_LOAD_FACTOR) {
			rehash();
		}
		return putHelp(k, v, keys, values);

	}

	public TIntOpenHashMap() {
		this(10);
	}

	public TIntOpenHashMap(int initialCapacity_) {
		this(initialCapacity_, 0.7);
	}

	public TIntOpenHashMap(int initialCapacity_, double loadFactor) {
		int cap = Math.max(5, (int) (initialCapacity_ / loadFactor));
		MAX_LOAD_FACTOR = loadFactor;
		values = new int[cap];
		Arrays.fill(values, -1);
		keys = (T[]) new Object[cap];
	}

	/**
	 * 
	 */
	private void rehash() {
		T[] newKeys = (T[]) new Object[keys.length * 3 / 2];
		int[] newValues = new int[values.length * 3 / 2];
		Arrays.fill(newValues, -1);
		size = 0;
		for (int i = 0; i < keys.length; ++i) {
			T curr = keys[i];
			if (curr != null) {
				int val = values[i];
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
	private boolean putHelp(T k, int v, T[] keyArray, int[] valueArray) {
		int pos = getInitialPos(k, keyArray);
		T curr = keyArray[pos];
		while (curr != null && !curr.equals(k)) {
			pos++;
			if (pos == keyArray.length) pos = 0;
			curr = keyArray[pos];
		}

		valueArray[pos] = v;
		if (curr == null) {
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
	private int getInitialPos(T k, T[] keyArray) {
		int hash = k.hashCode();
		int pos = hash % keyArray.length;
		if (pos < 0) pos += keyArray.length;
    // N.B. Doing it this old way causes Integer.MIN_VALUE to be
		// handled incorrect since -Integer.MIN_VALUE is still
		// Integer.MIN_VALUE
//		if (hash < 0) hash = -hash;
//		int pos = hash % keyArray.length;
		return pos;
	}

	public int get(T k) {
		int pos = find(k);

		return values[pos];
	}

	/**
	 * @param k
	 * @return
	 */
	private int find(T k) {
		int pos = getInitialPos(k, keys);
		T curr = keys[pos];
		while (curr != null && !curr.equals(k)) {
			pos++;
			if (pos == keys.length) pos = 0;
			curr = keys[pos];
		}
		return pos;
	}

	public static class Entry<T>
	{
		/**
		 * @param key
		 * @param value
		 */
		public Entry(T key, int value) {
			super();
			this.key = key;
			this.value = value;
		}

		public T key;

		public int value;

		public T getKey() {
			return key;
		}

		public int getValue() {
			return value;
		}
	}

	private class EntryIterator extends MapIterator<Entry<T>>
	{
		public Entry<T> next() {
			final int nextIndex = nextIndex();

			return new Entry<T>(keys[nextIndex], values[nextIndex]);
		}
	}

	private abstract class MapIterator<E> implements Iterator<E>
	{
		public MapIterator() {
			end = keys.length;
			next = -1;
			nextIndex();
		}

		public boolean hasNext() {
			return next < end;
		}

		int nextIndex() {
			int curr = next;
			do {
				next++;
			} while (next < end && keys[next] == EMPTY_KEY);
			return curr;
		}

		public void remove() {
			throw new UnsupportedOperationException();
		}

		private int next, end;
	}

	public Iterable<Entry<T>> entrySet() {
		return CollectionUtils.iterable(new EntryIterator());
	}

	public int size() {
		return size;
	}

}
