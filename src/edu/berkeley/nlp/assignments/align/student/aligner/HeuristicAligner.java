package edu.berkeley.nlp.assignments.align.student.aligner;

import edu.berkeley.nlp.assignments.align.student.util.BitPackingUtility;
import edu.berkeley.nlp.assignments.align.student.util.IntIntOpenHashMap;
import edu.berkeley.nlp.assignments.align.student.util.LongIntOpenHashMap;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Gorilla on 12/3/2016.
 */
public class HeuristicAligner implements WordAligner{
    private static StringIndexer wordIndexer = new StringIndexer();
    private static IntIntOpenHashMap enCounter = new IntIntOpenHashMap(3000000);
    private static IntIntOpenHashMap frCounter = new IntIntOpenHashMap(3000000);
    private static LongIntOpenHashMap frEnCounter = new LongIntOpenHashMap(50000000);

    public HeuristicAligner(Iterable<SentencePair> trainingData){
        System.out.println("Training for heuristic");
        int pairCount = 0;
        for (SentencePair pair: trainingData) {
            pairCount++;
            if (pairCount % 100000 == 0)
                System.out.println(pairCount + " sentences loaded");
            List<String> englishWords = pair.getEnglishWords();
            List<String> frenchWords = pair.getFrenchWords();

            List<Integer> enIdxList = new ArrayList<>();
            for (String en: englishWords) {
                // count word 'en'
                int enIdx = wordIndexer.addAndGetIndex(en);
                enCounter.increment(enIdx, 1);
                enIdxList.add(enIdx);
            }

            for (String fr: frenchWords) {
                // count word 'fr'
                int frIdx = wordIndexer.addAndGetIndex(fr);
                frCounter.increment(frIdx, 1);

                // count number of pair fr-en
                for (int enIdx: enIdxList) {
                    long bigramPack = BitPackingUtility.bitPackingBigram(frIdx, enIdx);
                    frEnCounter.increment(bigramPack, 1);
                }
            }
        }
        System.out.format("Training done with %d pair of sentences\n", pairCount);
        System.out.format("Size of enCounter=%d, frCounter=%d, frEnCounter=%d\n", enCounter.size(), frCounter.size(), frEnCounter.size());

        // debug
//        debugIntIntOpenHashMap(enCounter);
//        debugIntIntOpenHashMap(frCounter);
//        debugBigramPackingLongIntOpenHashMap(frEnCounter);
    }

    // most important method - will be called by tester
    @Override
    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignment = new Alignment();
        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();

        for (int j = 0; j < frenchWords.size(); j++) {
            String fr = frenchWords.get(j);
            int maxAlignedPosition = argmaxDiceCoefficient(fr, englishWords);
            // align this 'fr' word to the best 'en' word
            alignment.addAlignment(maxAlignedPosition, j, true);
        }

        return alignment;
    }

    /**
     * Constructor helper. Calculate maximum value based on Dice Coefficient 2c(f, e) / (c(f) + c(e))
     * @param fr: a french word
     * @param englishWords: a list of english words, for each word, we have a particular value of Dice Coefficient
     * @return: position of the english word, in the list, which yields the max Dice score
     */
    private int argmaxDiceCoefficient(String fr, List<String> englishWords) {
        int frIdx = wordIndexer.indexOf(fr);
        // align with an 'en' word <-> argmax Dice Coefficient
        double maxDiceCoeff = -1.;
        int maxAlignedPosition = 0; // set to 0 to avoid edge cases, where all denominator==0 -> align to the first word
        for (int i = 0; i < englishWords.size(); i++) {
            String en = englishWords.get(i);
            int enIdx = wordIndexer.indexOf(en);
            int diceDenominator = frCounter.get(frIdx) + enCounter.get(enIdx);

            if (diceDenominator != 0) {
                long bigramPack = BitPackingUtility.bitPackingBigram(frIdx, enIdx);
                double diceCoeff = (frEnCounter.get(bigramPack) / (float)diceDenominator);
                if (maxDiceCoeff <  diceCoeff){
                    maxDiceCoeff = diceCoeff;
                    maxAlignedPosition = i;
                }
            } // else continue
        }
        return maxAlignedPosition;
    }

    // ==================== DEBUG ===============================
    private void debugIntIntOpenHashMap(IntIntOpenHashMap map){
        for (IntIntOpenHashMap.Entry entry: map.entrySet()) {
            System.out.format("Key=%s Val=%d\n", wordIndexer.get(entry.getKey()), entry.getValue());
        }
    }
    private void debugBigramPackingLongIntOpenHashMap(LongIntOpenHashMap map) {
        for (LongIntOpenHashMap.Entry entry: map.entrySet()) {
            long key = entry.getKey();
            int[] decoded = BitPackingUtility.bigramBitPackingDecode(key);
            System.out.format("KeyPair= %s + %s Val=%d\n",wordIndexer.get(decoded[0]), wordIndexer.get(decoded[1]), entry.getValue());
        }
    }
}
