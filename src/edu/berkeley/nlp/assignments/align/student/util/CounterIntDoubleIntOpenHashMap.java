package edu.berkeley.nlp.assignments.align.student.util;

import edu.berkeley.nlp.util.CollectionUtils;

import java.util.Arrays;
import java.util.Iterator;

/**
 * key = index(e)
 * value1 = t(f|e)
 * value2 = count(f|e)
 */
public class CounterIntDoubleIntOpenHashMap {
    private int[] keys;

    private double[] values1;
    private int[] values2;

    private int size = 0;
    private int sizeInTheory = 0;
    private int actualSize = 0;


    private final int EMPTY_KEY = -1;

    private final double MAX_LOAD_FACTOR;

    // order: value -> end -> start -> between
    public boolean put(int k, double v, int end) {
        if (size / (double) keys.length > MAX_LOAD_FACTOR) {
            rehash();
        }
        return putHelp(k, v, end, keys, values1, values2);
    }

    public boolean putValue1(int k, double v) {
        if (size / (double) keys.length > MAX_LOAD_FACTOR) {
            rehash();
        }
        return putHelpValue1(k, v, keys, values1);
    }
    public boolean putValue2(int k, int e) {
        if (size / (double) keys.length > MAX_LOAD_FACTOR) {
            rehash();
        }
        return putHelpValue2(k, e, keys, values2);
    }

    public CounterIntDoubleIntOpenHashMap() {
        this(10);
    }

    public CounterIntDoubleIntOpenHashMap(int initialCapacity_) {
        this(initialCapacity_, 0.7);
    }

    public CounterIntDoubleIntOpenHashMap(int initialCapacity_, double loadFactor) {
        int cap = Math.max(5, (int) (initialCapacity_ / loadFactor));
        MAX_LOAD_FACTOR = loadFactor;

        values1 = new double[cap];
        values2 = new int[cap];
        Arrays.fill(values1, 0);
        Arrays.fill(values2, 0);

        keys = new int[cap];
        Arrays.fill(keys, -1); // added to avoid collision with k = 0

        sizeInTheory = initialCapacity_;
    }

    private void rehash() {
        int[] newKeys = new int[keys.length * 3 / 2];
        double[] newValues1 = new double[values1.length * 3 / 2];
        int[] newValues2 = new int[values1.length * 3 / 2];

        Arrays.fill(newValues1, 0);
        Arrays.fill(newValues2, 0);
        Arrays.fill(newKeys, -1);
        size = 0;
        for (int i = 0; i < keys.length; ++i) {
            int curr = keys[i];
            if (curr != EMPTY_KEY) {
                double val1 = values1[i];
                int val2 = values2[i];
                // TODO: fix this
                putHelp(curr, val1, val2, newKeys, newValues1, newValues2);
            }
        }
        // overwrite
        keys = newKeys;
        values1 = newValues1;
        values2 = newValues2;
    }
    public void rehash(double expandedRatio) {
        int[] newKeys = new int[(int) (keys.length * expandedRatio)];
        double[] newValues1 = new double[(int) (values1.length * expandedRatio)];
        int[] newValues2 = new int[(int) (values1.length * expandedRatio)];

        Arrays.fill(newValues1, 0);
        Arrays.fill(newValues2, 0);
        Arrays.fill(newKeys, -1);
        size = 0;
        for (int i = 0; i < keys.length; ++i) {
            int curr = keys[i];
            if (curr != EMPTY_KEY) {
                double val1 = values1[i];
                int val2 = values2[i];
                // TODO: fix this
                putHelp(curr, val1, val2, newKeys, newValues1, newValues2);
            }
        }
        // overwrite
        keys = newKeys;
        values1 = newValues1;
        values2 = newValues2;
    }

    // order: value -> end -> start -> between
    private boolean putHelp(int k, double val1, int val2,
                            int[] keyArray, double[] values1, int[] values2) {
        int pos = getInitialPos(k, keyArray);
        int curr = keyArray[pos];
        // find proper key first
        while (curr != EMPTY_KEY && curr != k) {
            pos++;
            if (pos == keyArray.length) pos = 0;
            curr = keyArray[pos];
        }

        values1[pos] = val1;
        values2[pos] = val2;

        // found a key, let's insert into it
        if (curr == EMPTY_KEY) {
            size++;
            keyArray[pos] = k;
            return true;
        }
        return false;
    }
    private boolean putHelpValue1(int k, double val1, int[] keyArray, double[] values1) {
        int pos = getInitialPos(k, keyArray);
        int curr = keyArray[pos];
        // find proper key first
        while (curr != EMPTY_KEY && curr != k) {
            pos++;
            if (pos == keyArray.length) pos = 0;
            curr = keyArray[pos];
        }

        values1[pos] = val1;
        // found a key, let's insert into it
        if (curr == EMPTY_KEY) {
            size++;
            keyArray[pos] = k;
            return true;
        }
        return false;
    }
    private boolean putHelpValue2(int k, int val2, int[] keyArray, int[] values2) {
        int pos = getInitialPos(k, keyArray);
        int curr = keyArray[pos];
        // find proper key first
        while (curr != EMPTY_KEY && curr != k) {
            pos++;
            if (pos == keyArray.length) pos = 0;
            curr = keyArray[pos];
        }

        values2[pos] = val2;
        // found a key, let's insert into it
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
//        int hash = ((int) (n ^ (n >>> 32)) * 3875239);
//        return hash;
    }

    // 4 getters
    public double getValue1(int k) {
        int pos = find(k);
        return values1[pos];
    }
    public int getValue2(int k) {
        int pos = find(k);
        return values2[pos];
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

    public void incrementValue1(int k, double v) {
        int pos = find(k);
        int currKey = keys[pos];
        // key is new
        if (currKey == EMPTY_KEY) {
            putValue1(k, v);
        } else
            values1[pos]+=v;
    }
    public void incrementValue2(int k, int e) {
        int pos = find(k);
        int currKey = keys[pos];
        // key is new
        if (currKey == EMPTY_KEY) {
            putValue2(k, e);
        } else {
            values2[pos]+=e;
        }
    }


    public static class Entry {
        /**
         * @param key
         * @param value1
         * @param value2
         */
        public Entry(int key, double value1, int value2) {
            super();
            this.key = key;
            this.value1 = value1;
            this.value2 = value2;
        }

        public int key;
        public double value1;
        public int value2;

        public int getKey() {
            return key;
        }
        public double getValue1() {
            return value1;
        }
        public int getValue2() {
            return value2;
        }
    }

    private class EntryIterator extends CounterIntDoubleIntOpenHashMap.MapIterator<CounterIntDoubleIntOpenHashMap.Entry>
    {
        public CounterIntDoubleIntOpenHashMap.Entry next() {
            final int nextIndex = nextIndex();
            //access arrays of values1 from mother class
            return new CounterIntDoubleIntOpenHashMap.Entry(keys[nextIndex], values1[nextIndex], values2[nextIndex]);
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

    public Iterable<CounterIntDoubleIntOpenHashMap.Entry> entrySet() {
        return CollectionUtils.iterable(new CounterIntDoubleIntOpenHashMap.EntryIterator());
    }
    public int[] getKeys() {
        int[] k = new int[size];
        int j = 0;
        for (int i = 0; i<keys.length; i++) {
            if (keys[i] != -1)
                k[j++] = keys[i];
        }
        return k;
    }
    public int size() {
        return size;
    }

    public int actualSize() {
        return keys.length;
    }


    /**
     * Optimization method to free up unused entries in this map
     * TODO: add values2 to this optimization before activating it
     */
//    public void optimizeStorage(double expandedRatio){
//        System.out.println("This map has the utilization of " + 100 * size() / (double) actualSize() + "%. Now optimizing...");
//
//        int[] newKeys = new int[size];
//        int[] newValues = new int[size];
//        int j = 0;
//
//        for (int i = 0; i< values1.length; i++) {
//            if (keys[i] != -1) {
//                newKeys[j] = keys[i];
//                newValues[j] = values1[i];
//                j++;
//            }
//        }
//        // free up
//        keys = newKeys;
//        values1 = newValues;
//
//        rehash(expandedRatio);
//    }
    public void autoOptimizeStorage() {
        double utilization = size / (double) actualSize();
        rehash(utilization + 0.2);
    }
}
