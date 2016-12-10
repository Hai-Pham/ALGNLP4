package edu.berkeley.nlp.assignments.align.student.util;

import java.util.Arrays;

/**
 * Created by Gorilla on 12/8/2016.
 */
public class ArrayUtils {
    /**
     * Take all rows, the specific columns from colIndex vectors
     * @param a: original array to be extracted
     * @param colIndex: indexes of columns to be taken
     * @return
     */
    public static float[][] extractColumsFromArray(float[][] a, int[] colIndex) {
        float[][] e = new float[a.length][colIndex.length];

        for (int rowIdx = 0; rowIdx < a.length; rowIdx++) {
            int j=0;
            for (int col: colIndex) {
                e[rowIdx][j++] = a[rowIdx][col];
            }
        }
        return e;
    }

    public static void print2DArray(float[][] a, String label) {
        System.out.println("Debugging content of " + label + " [");
        for (float[] row: a) {
            for (float f: row)
                System.out.print(f + ", ");
            System.out.println();
        }
        System.out.println("]");
    }

    public static void main(String[] args) {
        float[][] a = new float[3][4];

        int i = 0;
        for (float[] row: a) {
            Arrays.fill(row, i++);
        }
        a[1][1] = 5;
        a[2][2] = 6;
        print2DArray(a, "test");

        float[][] e = extractColumsFromArray(a, new int[] {0, 2});
        print2DArray(e, "test");
    }
}
