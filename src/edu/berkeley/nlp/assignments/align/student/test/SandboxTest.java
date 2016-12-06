package edu.berkeley.nlp.assignments.align.student.test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Created by Gorilla on 12/6/2016.
 */
public class SandboxTest {
    public static void main(String[] args) {
        List<Map<Integer, Double>> x = new ArrayList<>();

        int[][] a = new int[2][3];

        for (int[] aa : a)
            for (int aaa : aa)
                System.out.println(aaa);

        Arrays.fill(a[0], 6);
        for (int[] aa : a)
            for (int aaa : aa)
                System.out.println(aaa);

    }
}
