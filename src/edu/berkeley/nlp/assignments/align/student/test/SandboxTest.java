package edu.berkeley.nlp.assignments.align.student.test;

import edu.berkeley.nlp.math.SloppyMath;

import java.util.*;

/**
 * Created by Gorilla on 12/6/2016.
 */
public class SandboxTest {

    public static void print2DArray(float[][] a, String label) {
        System.out.println("Debugging content of " + label + " [");
        for (float[] row: a) {
            for (float f: row)
                System.out.print(f + ", ");
            System.out.println();
        }
        System.out.println("]");
    }
    private static float laplaceInit(float x, float mu, float b) {
        return (float) Math.exp(-Math.abs(x - mu) / b) / (2 * b);
    }
    public static void main(String[] args) {
        List<Map<Integer, Double>> x = new ArrayList<>();

        int[][] a = new int[2][3];
        float[][] af = new float[2][3];
        print2DArray(af, "test");

        for (int[] aa : a)
            for (int aaa : aa)
                System.out.println(aaa);

        Arrays.fill(a[0], 6);
        for (int[] aa : a)
            for (int aaa : aa)
                System.out.println(aaa);

        for (int i = 5; i >=0; i--)
            System.out.print(i + " ");
        System.out.println();


        // arraylist test
        List<List<List<Integer>>> V = new ArrayList<>();
        System.out.println(V);

        for (int i = 0; i < 5; i++)
            V.add(i, new ArrayList<>());
        System.out.println(V);

        List<Integer> l1 = new ArrayList<>();
        l1.add(5);
        l1.add(3);

        // list copy
        List<Integer> l2 = new ArrayList<>(l1);
//        Collections.copy(l2, l1);
        System.out.println(l2);
        System.out.println(l2 == l1);

        l2.add(7);
        System.out.println(l2);
        System.out.println(l1);

        int[] i = new int[] {2, 3, 4, 5, 6};

        System.out.println(laplaceInit(1, 1, 2));
        System.out.println(laplaceInit(0, 1, 2));
        System.out.println(laplaceInit(2, 1, 2));
        System.out.println(laplaceInit(5, 1, 2));
        System.out.println(laplaceInit(10, 1, 2));

        System.out.println(SloppyMath.logAdd(-11.0, -1.0));
        System.out.println(SloppyMath.logAdd(-11.0, Double.NEGATIVE_INFINITY));
    }
}
