package edu.berkeley.nlp.assignments.align.student.aligner;

import edu.berkeley.nlp.assignments.align.student.util.*;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Gorilla on 12/3/2016.
 */
public class IBM1Aligner implements WordAligner {

    private static StringIndexer enIndexer = new StringIndexer();
    private static StringIndexer frIndexer = new StringIndexer();

//    private static LongDoubleOpenHashMap probFoverE = new LongDoubleOpenHashMap(5000000);
    private static MyIntegerDoubleCompoundHashMap probFoverE = new MyIntegerDoubleCompoundHashMap();
//    private static double[][] probFoverE ;


//    private static Map<Integer, Map<Integer, Double>> probEoverF = new HashMap<>();

    public IBM1Aligner(Iterable<SentencePair> trainingData) {
        System.out.println("Training for IBM Model 1");
        int pairCount = initializeEM(trainingData);

        // convergence criteria
        int targetNumIterations = 10;
        double delta = .001;
        int iter = 0;
        while (iter < targetNumIterations) {
            System.out.println("\n-----------------\nIteration " + iter + "\n-----------------\n");

            // init count(f|e) and total(e) for all e, f
            // TODO: just init once and use for all loops
//            LongDoubleOpenHashMap countFoverE = new LongDoubleOpenHashMap(200000);
//            IntDoubleOpenHashMap totalE = new IntDoubleOpenHashMap(20000);
            MyIntegerDoubleCompoundHashMap countFoverE = new MyIntegerDoubleCompoundHashMap(); // int int double
            MyIntegerDoubleHashMap totalE = new MyIntegerDoubleHashMap(); // int double

//            debugIntDoubleOpenHashMap(totalE); // OK
//            debugBigramPackingLongDoubleOpenHashMap(countFoverE); // OK

            int count = 0; // count number of pairs
            for (SentencePair pair : trainingData) {
                if (count % 100 == 0)
                    System.out.println(count + " pairs processed");

                List<String> englishWords = pair.getEnglishWords();
                englishWords.add(0, "NULL"); //add NULL token
                List<String> frenchWords = pair.getFrenchWords();
//                System.out.println("================\nReading sentence pair " + count + " En: " + englishWords + " Fr: " + frenchWords);

                // compute normalization factor for each pair
                MyIntegerDoubleHashMap currentSentenceTotalF = new MyIntegerDoubleHashMap();
                List<Integer> frIdxList = new ArrayList<>();
                List<Integer> enIdxList = new ArrayList<>();
                List<Double> tF_over_E_List = new ArrayList<>();
                boolean isEnListAdded = true;
                for (String fr : frenchWords) {
//                    System.out.print("fr=" + fr + " en= ");
                    int frIdx = frIndexer.indexOf(fr);
                    frIdxList.add(frIdx);
                    for (String en: englishWords) {
//                        System.out.print(en + " ");
                        int enIdx = enIndexer.indexOf(en);
                        if (isEnListAdded) enIdxList.add(enIdx);

                        double t_F_over_E = 0.;
                        if (iter ==0) {
                            probFoverE.put(frIdx, new MyIntegerDoubleHashMap());

//                            t_F_over_E = 1./frIndexer.size();
                            if (en == "NULL")
                                t_F_over_E += .2;
                            else
                                t_F_over_E += .8/frIndexer.size();//
                            probFoverE.get(frIdx).put(enIdx, t_F_over_E);
                        } else
                            t_F_over_E = probFoverE.get(frIdx).get(enIdx);

//                        System.out.print(" t(f|e)=" + t_F_over_E + ", ");
                        tF_over_E_List.add(t_F_over_E);
                        currentSentenceTotalF.increment(frIdx, t_F_over_E);
                    }
//                    System.out.println();
                    isEnListAdded = false; // next loop will not redundantly add into enIdxList
                }

                // debug
//                System.out.println("-----\nDebugging normalization factor");
//                System.out.println(currentSentenceTotalF);
//                System.out.println("Debug 3 lists: frIdxlist, enIdxList, tF_over_E_List");
//                System.out.println(frIdxList);
//                System.out.println(enIdxList);
//                System.out.println(tF_over_E_List);

                // collect counts
                int c = 0;
                for (int frIdx: frIdxList) {
                    for (int enIdx: enIdxList) {
                        double t_F_over_E = tF_over_E_List.get(c);
                        double sTotalF = currentSentenceTotalF.get(frIdx);
//                        if (!countFoverE.containsKey(frIdx))
//                            System.out.println("frIdx=" + frIdx + " enIdx=" + enIdx + " t(f|e)=" + t_F_over_E + " s-total(f)=" + sTotalF + " c(f|e)=0" + " totalE=" + totalE.get(enIdx));
//                        else
//                            System.out.println("frIdx=" + frIdx + " enIdx=" + enIdx + " t(f|e)=" + t_F_over_E + " s-total(f)=" + sTotalF + " c(f|e)=" + countFoverE.get(frIdx).get(enIdx) + " totalE=" + totalE.get(enIdx));
                        countFoverE.increment(frIdx, enIdx, t_F_over_E / sTotalF);

//                        System.out.println("new c(" + frIndexer.get(frIdx) + "|" + enIndexer.get(enIdx) + ")=" + countFoverE.get(frIdx).get(enIdx));
                        totalE.increment(enIdx, t_F_over_E / sTotalF);
//                        System.out.println("new total(" + enIndexer.get(enIdx) + ")=" + totalE.get(enIdx) + "\n");
                        c++;
                    }
                }
                englishWords.remove(0); // remove token NULL
                count ++;
            }

            // now reestimate master t(f|e) after processing all training samples
            System.out.println("Reupdating t(f|e)");
            for (int f = 0; f < frIndexer.size(); f++) {
                MyIntegerDoubleHashMap currentProbFoverEMap = probFoverE.get(f);
                MyIntegerDoubleHashMap currentCountFoverEMap = countFoverE.get(f);
                for (int e = 0; e < enIndexer.size(); e++) {
                    currentProbFoverEMap.put(e, currentCountFoverEMap.get(e) / totalE.get(e));
                }
            }

            // update num iters
            iter++;
//            countFoverE = null;
//            totalE = null;
//            System.gc();
        }
    }


    private int initializeEM(Iterable<SentencePair> trainingData) {

        // initialization
        System.out.println("Scanning through all data and initializing EM...");
        System.out.println("Index of NULL token is " + enIndexer.addAndGetIndex("NULL")); // NULL TOKEN for ENGLISH

        int pairCount = 0;
        for (SentencePair pair: trainingData) {
            pairCount++;
            if (pairCount % 10000 == 0) System.out.println(pairCount + " sentences loaded");
            List<String> englishWords = pair.getEnglishWords();
            List<String> frenchWords = pair.getFrenchWords();

            // get english words
            List<Integer> enIdxList = new ArrayList<>();
            for (String en: englishWords) {
                int enIdx = enIndexer.addAndGetIndex(en);
                enIdxList.add(enIdx);
            }
            // and scan thru french words
            List<Integer> frIdxList = new ArrayList<>();
            for (String fr: frenchWords) {
                int frIdx = frIndexer.addAndGetIndex(fr);
                frIdxList.add(frIdx);
            }
        }
        // uniformly init probability
        int frVocabSize = frIndexer.size();
        System.out.println("Size of FR vocab is " + frVocabSize + " size of EN vocab is " + enIndexer.size());
        System.out.println("Initializing done!");
        return pairCount;
    }


    // ============ most important method, will be called by tester ==============
    @Override
    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignment = new Alignment();

        List<String> englishWords = sentencePair.getEnglishWords();
//        englishWords.add(0, "NULL");
        List<String> frenchWords = sentencePair.getFrenchWords();
        for (int j = 0; j < frenchWords.size(); j++) {
            int frIdx = frIndexer.indexOf(frenchWords.get(j));

//            double maxProb = probFoverE.get(BitPackingUtility.bitPackingBigram(frIdx, 0)); // set max to fr-> NULL
            double maxProb = probFoverE.get(frIdx).get(0);
            int bestPosition = -1; // NULL position
            for (int i = 0; i < englishWords.size(); i++) {
                int enIdx = enIndexer.indexOf(englishWords.get(i));
                long frEnIdx = BitPackingUtility.bitPackingBigram(frIdx, enIdx);

//                double t_F_over_E = probFoverE.get(frEnIdx);
                double t_F_over_E = probFoverE.get(frIdx).get(enIdx);
                if (maxProb < t_F_over_E) {
                    maxProb = t_F_over_E;
                    bestPosition = i; // we already count NULL position = 0
                }
            }
            // align
//            System.out.println("Aligned " + frenchWords.get(j) + " to " + englishWords.get(bestPosition));
            alignment.addAlignment(bestPosition, j, true); // if add NULL to english sentence then (--bestPosition)
        }
//        englishWords.remove(0);
        return alignment;
    }

    // ==================== DEBUG ===============================
    private void debugIntIntOpenHashMap(IntIntOpenHashMap map){
        System.out.println("\nContent of IntIntOpenHashMap with enIndexer");
        for (IntIntOpenHashMap.Entry entry: map.entrySet()) {
            System.out.format("Key=%s\tVal=%d\n", enIndexer.get(entry.getKey()), entry.getValue());
        }
        System.out.println("-----------------");
    }
    private void debugIntDoubleOpenHashMap(IntDoubleOpenHashMap map){
        System.out.println("\nContent of IntDoubleOpenHashMap with frIndexer");
        for (IntDoubleOpenHashMap.Entry entry: map.entrySet()) {
            System.out.format("Key=%s\tVal=%.4f\n", frIndexer.get(entry.getKey()), entry.getValue());
        }
        System.out.println("-----------------");
    }
    private void debugLongDoubleOpenHashMap(LongDoubleOpenHashMap map){
        System.out.println("\nContent of IntDoubleOpenHashMap with frIndexer");
        for (LongDoubleOpenHashMap.Entry entry: map.entrySet()) {
            long key = entry.getKey();
            int[] decoded = BitPackingUtility.bigramBitPackingDecode(key);
            System.out.format("KeyPair= (%s + %s)\tVal=%d\n", frIndexer.get(decoded[0]), enIndexer.get(decoded[1]), entry.getValue());
        }
        System.out.println("-----------------");
    }
    private void debugBigramPackingLongIntOpenHashMap(LongIntOpenHashMap map) {
        System.out.println("\nContent of LongIntOpenHashMap");
        for (LongIntOpenHashMap.Entry entry: map.entrySet()) {
            long key = entry.getKey();
            int[] decoded = BitPackingUtility.bigramBitPackingDecode(key);
            System.out.format("KeyPair= (%s + %s)\tVal=%d\n", frIndexer.get(decoded[0]), enIndexer.get(decoded[1]), entry.getValue());
        }
        System.out.println("-----------------");
    }
    private void debugBigramPackingLongDoubleOpenHashMap(LongDoubleOpenHashMap map) {
        System.out.println("\nContent of Long Double OpenHashMap");
        for (LongDoubleOpenHashMap.Entry entry: map.entrySet()) {
            long key = entry.getKey();
            int[] decoded = BitPackingUtility.bigramBitPackingDecode(key);
            System.out.format("KeyPair=%d + %d (%s + %s)\tVal=%.4f\n",decoded[0], decoded[1], frIndexer.get(decoded[0]), enIndexer.get(decoded[1]), entry.getValue());
        }
        System.out.println("-----------------");
    }
    private void debugStringIndexer(StringIndexer s, String name) {
        System.out.println("Debug string indexer: " + name);
        for (int i = 0; i < s.size(); i++) {
            System.out.format("k=%d, val=%s\n", i, s.get(i));
        }
    }
}
