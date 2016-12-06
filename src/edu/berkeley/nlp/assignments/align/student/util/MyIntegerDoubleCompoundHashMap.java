package edu.berkeley.nlp.assignments.align.student.util;

import java.util.HashMap;

/**
 * Created by Gorilla on 12/6/2016.
 */
public class MyIntegerDoubleCompoundHashMap extends HashMap<Integer, MyIntegerDoubleHashMap> {

    public void increment(int key1, int key2, double value) {
        if (!this.containsKey(key1)) {
            this.put(key1, new MyIntegerDoubleHashMap());
        }

        this.get(key1).increment(key2, value);
    }

    public static void main(String[] args) {
        MyIntegerDoubleCompoundHashMap map = new MyIntegerDoubleCompoundHashMap();

        map.increment(1, 2, 3.0);

        System.out.println(map.get(1).get(2));

        System.out.println(map);
    }
}
