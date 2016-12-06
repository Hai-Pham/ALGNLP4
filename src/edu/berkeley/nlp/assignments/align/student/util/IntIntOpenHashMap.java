package edu.berkeley.nlp.assignments.align.student.util;

import edu.berkeley.nlp.util.CollectionUtils;

import java.util.Arrays;
import java.util.Iterator;


public class IntIntOpenHashMap {

    private int[] keys;

    private int[] values;

    private int size = 0;
    private int sizeInTheory = 0;
    private int actualSize = 0;

    private final int EMPTY_KEY = -1;

    private final double MAX_LOAD_FACTOR;

    public boolean put(int k, int v) {
        if (size / (double) keys.length > MAX_LOAD_FACTOR) {
            rehash();
        }
        return putHelp(k, v, keys, values);

    }

    public IntIntOpenHashMap() {
        this(10);
    }

    public IntIntOpenHashMap(int initialCapacity_) {
        this(initialCapacity_, 0.7);
    }

    public IntIntOpenHashMap(int initialCapacity_, double loadFactor) {
        int cap = Math.max(5, (int) (initialCapacity_ / loadFactor));
        MAX_LOAD_FACTOR = loadFactor;
        values = new int[cap];
        Arrays.fill(values, -1);
        keys = new int[cap];
        Arrays.fill(keys, -1); // added to avoid collision with k = 0
        sizeInTheory = initialCapacity_;
    }

    private void rehash() {
        int[] newKeys = new int[keys.length * 3 / 2];
        int[] newValues = new int[values.length * 3 / 2];
        Arrays.fill(newValues, -1);
        Arrays.fill(newKeys, -1);
        size = 0;
        for (int i = 0; i < keys.length; ++i) {
            int curr = keys[i];
            if (curr != EMPTY_KEY) {
                int val = values[i];
                putHelp(curr, val, newKeys, newValues);
            }
        }
        keys = newKeys;
        values = newValues;
    }

    public void rehash(double expandedRatio) {
        int[] newKeys = new int[ (int)(keys.length * expandedRatio)];
        int[] newValues = new int[(int) (values.length * expandedRatio)];
        Arrays.fill(newValues, -1);
        Arrays.fill(newKeys, -1);
        size = 0;
        for (int i = 0; i < keys.length; ++i) {
            int curr = keys[i];
            if (curr != EMPTY_KEY) {
                int val = values[i];
                putHelp(curr, val, newKeys, newValues);
            }
        }
        keys = newKeys;
        values = newValues;
    }

    private boolean putHelp(int k, int v, int[] keyArray, int[] valueArray) {
        int pos = getInitialPos(k, keyArray);
        int curr = keyArray[pos];
        while (curr != EMPTY_KEY && curr != k) {
            pos++;
            if (pos == keyArray.length) pos = 0;
            curr = keyArray[pos];
        }

        valueArray[pos] = v;
        if (curr == EMPTY_KEY) {
            size++;
            keyArray[pos] = k;
            return true;
        }
        return false;
    }

    private int getInitialPos(int k, int[] keyArray) {
        int hash = getHashCode(k);
        int pos = hash % keyArray.length;
        if (pos < 0) pos += keyArray.length;
        // N.B. Doing it this old way causes Integer.MIN_VALUE to be
        // handled incorrect since -Integer.MIN_VALUE is still
        // Integer.MIN_VALUE
//		if (hash < 0) hash = -hash;
//		int pos = hash % keyArray.length;
        return pos;
    }
    // helper for hash code
    private int getHashCode(int n) {
        return (int) ((131111L*n)^n^(1973*n)%sizeInTheory);
    }

    public int get(int k) {
        int pos = find(k);

        return values[pos];
    }

    private int find(int k) {
        int pos = getInitialPos(k, keys);
        int curr = keys[pos];
        while (curr != EMPTY_KEY && curr != k) {
            pos++;
            if (pos == keys.length) pos = 0;
            curr = keys[pos];
        }
        return pos;
    }

    public void increment(int k, int c) {
        int pos = find(k);
        int currKey = keys[pos];
        if (currKey == EMPTY_KEY) {
            put(k, c);
        } else
            values[pos]+=c;
    }

    public void incrementByOne(int k) {
        int pos = find(k);
        int currKey = keys[pos];
        if (currKey == EMPTY_KEY) {
            put(k, 1);
        } else
            values[pos]++;
    }

    public static class Entry
    {
        /**
         * @param key
         * @param value
         */
        public Entry(int key, int value) {
            super();
            this.key = key;
            this.value = value;
        }

        public int key;

        public int value;

        public int getKey() {
            return key;
        }

        public int getValue() {
            return value;
        }
    }

    private class EntryIterator extends MapIterator<Entry>
    {
        public Entry next() {
            final int nextIndex = nextIndex();

            return new Entry(keys[nextIndex], values[nextIndex]);
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

    public Iterable<Entry> entrySet() {
        return CollectionUtils.iterable(new EntryIterator());
    }

    public int size() {
        return size;
    }

    public int actualSize() {
        return keys.length;
    }
    /**
     * Optimization method to free up unused entries in this map
     *
     */
    public void optimizeStorage(double expandedRatio){
        System.out.println("This map has the utilization of " + 100 * size() / (double) actualSize() + "%. Now optimizing...");

        int[] newKeys = new int[size];
        int[] newValues = new int[size];
        int j = 0;

        for (int i=0; i<values.length; i++) {
            if (keys[i] != -1) {
                newKeys[j] = keys[i];
                newValues[j] = values[i];
                j++;
            }
        }
        // free up
        keys = newKeys;

        rehash(expandedRatio);
    }
}
