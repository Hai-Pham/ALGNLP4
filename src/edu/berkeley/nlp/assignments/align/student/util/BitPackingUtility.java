package edu.berkeley.nlp.assignments.align.student.util;

/**
 * Created by Gorilla on 12/3/2016.
 */
import java.util.*;

public class BitPackingUtility {
    /**
     * Bit packing for bigram
     * @param w1
     * @param w2
     * @return bit packing as a long (64-bit) number with 32 bits for w2, 32 for w1
     * The format is w2w1 with w2 is 32-bit context
     */

    public static long bitPackingBigram(int w1, int w2) {
        long lw2 = ((long) w2) << 32;
        long lw1 = w1 & 0xFFFFFFFFL;

        return lw2 | lw1;
    }

    /**
     * Decode bitpacking for Bigram
     * @param encodedPacking
     * @return an array storing the numbers for w1, w2 respectively
     * Note: the order is w2w1
     */
    public static int[] bigramBitPackingDecode(long encodedPacking) {
        int[] result = new int[2];

        result[0] = (int)(encodedPacking & 0xFFFFFFFFL); //w1
        result[1] = (int)(encodedPacking >> 32); //w2

        return result;
    }
}