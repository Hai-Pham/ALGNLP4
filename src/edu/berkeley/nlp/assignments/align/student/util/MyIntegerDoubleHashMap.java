package edu.berkeley.nlp.assignments.align.student.util;

import java.util.HashMap;

/**
 * Created by Gorilla on 12/6/2016.
 */
public class MyIntegerDoubleHashMap extends HashMap<Integer, Double> {
    @Override
    public Double get(Object key) {
        if (!this.containsKey(key))
            return 0.;
        return super.get(key);
    }

    public void increment(int key, double val) {
        Double current = this.get(key);
        this.put(key, current + val);
    }

    public static void main(String[] args) {
        MyIntegerDoubleHashMap map = new MyIntegerDoubleHashMap();
        System.out.println(map.get(1));
        map.increment(1, 4);
        System.out.println(map.get(1));
    }
}
