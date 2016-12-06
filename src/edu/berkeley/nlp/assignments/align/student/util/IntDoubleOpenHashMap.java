package edu.berkeley.nlp.assignments.align.student.util;

import edu.berkeley.nlp.util.CollectionUtils;

import java.util.Arrays;
import java.util.Iterator;

public class IntDoubleOpenHashMap {

    private int[] keys;
    private double[] values;

    private int size = 0;
    private int sizeInTheory = 0;
    private int actualSize = 0;

    private final long EMPTY_KEY = -1;
    private final double MAX_LOAD_FACTOR;

    public boolean put(int k, double v) {
        if (size / (double) keys.length > MAX_LOAD_FACTOR) {
            rehash();
        }
        return putHelp(k, v, keys, values);

    }

    public IntDoubleOpenHashMap() {
        this(10);
    }

    public IntDoubleOpenHashMap(int initialCapacity_) {
        this(initialCapacity_, 0.7);
    }

    public IntDoubleOpenHashMap(int initialCapacity_, double loadFactor) {
        int cap = Math.max(5, (int) (initialCapacity_ / loadFactor));
        MAX_LOAD_FACTOR = loadFactor;
        values = new double[cap];
        Arrays.fill(values, 0);
        keys = new int[cap];
        Arrays.fill(keys, -1); // added to avoid collision with k = 0
        sizeInTheory = initialCapacity_;
    }


    private void rehash() {
        int[] newKeys = new int[keys.length * 3 / 2];
        double[] newValues = new double[values.length * 3 / 2];
        Arrays.fill(newValues, 0);
        Arrays.fill(newKeys, -1);
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
    public void rehash(double expandedRatio) {
        int[] newKeys = new int[(int)(keys.length * expandedRatio)];
        double[] newValues = new double[(int)(values.length * expandedRatio)];
        Arrays.fill(newValues, 0);
        Arrays.fill(newKeys, -1);
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
    private boolean putHelp(int k, double v, int[] keyArray, double[] valueArray) {
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
//        int hash = getHashCode(k);
        Integer kk = (Integer) k;
        int hash = kk.hashCode();
        int pos = (int) (hash % keyArray.length);
        if (pos < 0) pos += keyArray.length;
        // N.B. Doing it this old way causes Integer.MIN_VALUE to be
        // handled incorrect since -Integer.MIN_VALUE is still
        // Integer.MIN_VALUE
//		if (hash < 0) hash = -hash;
//		int pos = hash % keyArray.length;
        return pos;
    }
    // helper for hash code
    private int getHashCode(long n) {
        return (int)((131111L*n)^n^(1973*n)%sizeInTheory);
//        int hash = ((int) (n ^ (n >>> 32)) * 3875239);
//        return hash%sizeInTheory;
    }

    public double get(int k) {
        int pos = find(k);
        return values[pos];
    }

    private int find(int k) {
        int pos = getInitialPos(k, keys);
        long curr = keys[pos];
        while (curr != EMPTY_KEY && curr != k) {
            pos++;
            if (pos == keys.length) pos = 0;
            curr = keys[pos];
        }
        return pos;
    }

    public void increment(int k, double c) {
        int pos = find(k);
        long currKey = keys[pos];
        if (currKey == EMPTY_KEY) {
            put(k, c);
        } else
            values[pos]+=c;
    }
    public void incrementByOne(int k) {
        int pos = find(k);
        long currKey = keys[pos];
        if (currKey == EMPTY_KEY) {
            put(k, 1);
        } else
            values[pos]++;
    }
    public static class Entry
    {
        public Entry(int key, double value) {
            super();
            this.key = key;
            this.value = value;
        }

        public int key;
        public double value;
        public int getKey() {
            return key;
        }
        public double getValue() {
            return value;
        }
    }

    private class EntryIterator extends MapIterator<IntDoubleOpenHashMap.Entry> {
        public IntDoubleOpenHashMap.Entry next() {
            final int nextIndex = nextIndex();
            return new IntDoubleOpenHashMap.Entry(keys[nextIndex], values[nextIndex]);
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

    public Iterable<IntDoubleOpenHashMap.Entry> entrySet() {
        return CollectionUtils.iterable(new IntDoubleOpenHashMap.EntryIterator());
    }

    public int size() {
        return size;
    }

    public int actualSize() {
        return keys.length;
    }

//    /**
//     * Optimization method to free up unused entries in this map
//     *
//     */
//    public void optimizeStorage(double expandedRatio){
//        System.out.println("This map has the utilization of " + 100 * size / (float) keys.length + "%. Now optimizing...");
//
//        long[] newKeys = new long[size];
//        int[] newValues = new int[size];
//        int j = 0;
//
//        for (int i=0; i<values.length; i++) {
//            if (values[i] != 0) {
//                newKeys[j] = keys[i];
//                newValues[j] = values[i];
//                j++;
//            }
//        }
//        // free up
//        keys = newKeys;
//        values = newValues;
//
//        rehash(expandedRatio);
//    }
}