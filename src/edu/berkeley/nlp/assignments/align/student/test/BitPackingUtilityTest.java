package edu.berkeley.nlp.assignments.align.student.test;

import edu.berkeley.nlp.assignments.align.student.util.BitPackingUtility;

/**
 * Created by Gorilla on 12/3/2016.
 */
public class BitPackingUtilityTest {
    public static void main(String[] args) {
        int w1 = 12345;
        int w2 = 4324837;

        long bigramPack = BitPackingUtility.bitPackingBigram(w1, w2);

        int[] decoded = BitPackingUtility.bigramBitPackingDecode(bigramPack);
        System.out.println(decoded[0]); // w1
        System.out.println(decoded[1]); // w2
    }
}
