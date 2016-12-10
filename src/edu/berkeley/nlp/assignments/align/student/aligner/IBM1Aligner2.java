package edu.berkeley.nlp.assignments.align.student.aligner;

import edu.berkeley.nlp.assignments.align.student.util.*;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.*;

/**
 * Created by Gorilla on 12/3/2016.
 */
public class IBM1Aligner2 implements WordAligner {

    private static StringIndexer enIndexer = new StringIndexer();
    private static StringIndexer frIndexer = new StringIndexer();
    private static float[][] probFoverE;
    private static float[][] probEoverF;

    // CONSTRUCTOR
    public IBM1Aligner2(Iterable<SentencePair> trainingData) {
        System.out.println("Training for IBM Model 1");
        initializeEM(trainingData);

        // convergence criteria
        int targetNumIterations = 50;
        double delta = 1e-7;
        forwardModel1(trainingData, targetNumIterations, delta);
        backwardModel1(trainingData, targetNumIterations, delta);
    }

    // ============ INITIALIZATION ==============
    private void initializeEM(Iterable<SentencePair> trainingData) {
        // initialization
        System.out.println("Scanning through all data and initializing EM...");
        System.out.println("Index of NULL token is " + enIndexer.addAndGetIndex("null")); // NULL TOKEN for ENGLISH
        frIndexer.add("null");

        int pairCount = 0;
        for (SentencePair pair : trainingData) {
            pairCount++;
            if (pairCount % 1000 == 0) System.out.println(pairCount + " sentences loaded");
            List<String> englishWords = pair.getEnglishWords();
            List<String> frenchWords = pair.getFrenchWords();
            // get english words
            for (String en : englishWords)
                enIndexer.add(en.toLowerCase());
            // and scan thru french words
            for (String fr : frenchWords)
                frIndexer.add(fr.toLowerCase());
        }
        // uniformly init probability
        int frVocabSize = frIndexer.size();
        int enVocabSize = enIndexer.size();
        System.out.println("Size of FR vocab is " + frVocabSize + " size of EN vocab is " + enVocabSize);
        System.out.println("Initializing done!");

        // init 2D arrays
        probFoverE = new float[frVocabSize][enVocabSize];
        probEoverF = new float[enVocabSize][frVocabSize];
    }

    // ============ FORWARD MODEL1 ==============
    private void forwardModel1(Iterable<SentencePair> trainingData, int targetNumIterations, double delta) {
        int iter = 0;
        double lossFW = 1.0;
        while ((iter < targetNumIterations) && (lossFW > delta)) {
            System.out.println("\n-----------------\nFORWARD Iteration " + iter + "\n-----------------\n");

            // init count(f|e) and total(e) for all e, f
            float[][] countFoverE = new float[frIndexer.size()][enIndexer.size()];
            float[] totalE = new float[enIndexer.size()];

            int count = 0; // count number of pairs
            for (SentencePair pair : trainingData) {
                if (count % 1000 == 0)
                    System.out.println(count + " pairs processed");

                List<String> englishWords = pair.getEnglishWords();
                List<String> frenchWords = pair.getFrenchWords();

                // compute normalization factor for each pair
                MyIntegerDoubleHashMap sentenceTotalF = new MyIntegerDoubleHashMap(); // sparse, so use hashmap
                List<Integer> frIdxFwdList = new ArrayList<>();
                List<Integer> enIdxFwdList = new ArrayList<>();
                List<Float> tF_over_E_FwdList = new ArrayList<>();

                calculateSentenceNormalizationForward(iter, englishWords, frenchWords, sentenceTotalF, frIdxFwdList, enIdxFwdList, tF_over_E_FwdList);
                calculatePartialCountForward(countFoverE, totalE, sentenceTotalF, frIdxFwdList, enIdxFwdList, tF_over_E_FwdList);

                // count number of training pairs
                sentenceTotalF = null;
                count++;
            }

            lossFW = updateProbabilitiesAndLossForward(countFoverE, totalE);

            countFoverE = null;
            totalE = null;
            iter++;
        }
    }

    private void calculateSentenceNormalizationForward(int iter, List<String> englishWords, List<String> frenchWords, MyIntegerDoubleHashMap sentenceTotalF, List<Integer> frIdxFwdList, List<Integer> enIdxFwdList, List<Float> tF_over_E_FwdList) {
        //====================== FORWARD t(f|e) ======================
        englishWords.add(0, "null"); //add NULL token
        boolean isEnFwdListAdded = true;
        for (String fr : frenchWords) {
            int frIdx = frIndexer.indexOf(fr.toLowerCase());
            frIdxFwdList.add(frIdx);
            for (String en : englishWords) {
                int enIdx = enIndexer.indexOf(en.toLowerCase());
                if (isEnFwdListAdded) enIdxFwdList.add(enIdx); // add 0 at first

                float t_F_over_E = 0;
                if (iter == 0) {
//                    if (en.equals("null"))
//                        t_F_over_E += .2;
//                    else
//                        t_F_over_E += .8 / frIndexer.size();//
                    t_F_over_E += 1. / frIndexer.size();
                    probFoverE[frIdx][enIdx] = t_F_over_E;
                } else
                    t_F_over_E = probFoverE[frIdx][enIdx];

                tF_over_E_FwdList.add(t_F_over_E); // add t(F|NULL) for all F at first
                sentenceTotalF.increment(frIdx, t_F_over_E);
            }
            isEnFwdListAdded = false; // next loop will not redundantly add into enIdxList
        }
        englishWords.remove(0); // remove token NULL
    }

    private void calculatePartialCountForward(float[][] countFoverE, float[] totalE, MyIntegerDoubleHashMap sentenceTotalF, List<Integer> frIdxFwdList, List<Integer> enIdxFwdList, List<Float> tF_over_E_FwdList) {
        // collect partial counts
        int c = 0;
        for (int frIdx : frIdxFwdList) {
            for (int enIdx : enIdxFwdList) {
                float t_F_over_E = tF_over_E_FwdList.get(c);
                double s_total_F = sentenceTotalF.get(frIdx);
                countFoverE[frIdx][enIdx] += t_F_over_E / s_total_F;
                totalE[enIdx] += t_F_over_E / s_total_F;
                c++;
            }
        }
    }

    private double updateProbabilitiesAndLossForward(float[][] countFoverE, float[] totalE) {
        double lossFW;//==================MASTER UPDATING ====================
        // now reestimate master t(f|e) after processing all training samples
        System.out.println("Reupdating t(f|e)");
        lossFW = 0;
        for (int f = 0; f < frIndexer.size(); f++) {
            float[] currentProbFoverEMap = probFoverE[f];
            float[] currentCountFoverEMap = countFoverE[f];
            for (int e = 0; e < enIndexer.size(); e++) {
//                    currentProbFoverEMap[e] = currentCountFoverEMap[e] / totalE[e];
                float newVal = currentCountFoverEMap[e] / totalE[e];
                lossFW += Math.abs(currentProbFoverEMap[e] - newVal);
                currentProbFoverEMap[e] = newVal;
            }
        }
        // normalizing loss
        lossFW /= (enIndexer.size() * frIndexer.size());
        return lossFW;
    }

    // ============ BACKWARD MODEL1 ==============
    private void backwardModel1(Iterable<SentencePair> trainingData, int targetNumIterations, double delta) {
        int iter;// ==================BACKWARD================================================
        iter = 0;
        double lossBW = 1;
        while ((iter < targetNumIterations) && (lossBW > delta)) {
            System.out.println("\n-----------------\nBACKWARD Iteration " + iter + "\n-----------------\n");
            float[][] countEoverF = new float[enIndexer.size()][frIndexer.size()]; // backward
            float[] totalF = new float[frIndexer.size()]; // backward
            int count = 0; // count number of pairs
            for (SentencePair pair : trainingData) {
                if (count % 1000 == 0)
                    System.out.println(count + " pairs processed");

                List<String> englishWords = pair.getEnglishWords();
                List<String> frenchWords = pair.getFrenchWords();
                MyIntegerDoubleHashMap sentenceTotalE = new MyIntegerDoubleHashMap();
                List<Integer> frIdxBwdList = new ArrayList<>();
                List<Integer> enIdxBwdList = new ArrayList<>();
                List<Float> tE_over_F_BwdList = new ArrayList<>();


                calculateSentenceNormalizationBackward(iter, englishWords, frenchWords, sentenceTotalE, frIdxBwdList, enIdxBwdList, tE_over_F_BwdList);
                calculatePartialCountsBackward(countEoverF, totalF, sentenceTotalE, frIdxBwdList, enIdxBwdList, tE_over_F_BwdList);


                // count number of training pairs
                sentenceTotalE = null;
                count++;
            }

            lossBW = updateProbabilitiesAndLossBackward(countEoverF, totalF);

            // update num iters
            countEoverF = null;
            totalF = null;
            iter++;
        }
    }

    private void calculateSentenceNormalizationBackward(int iter, List<String> englishWords, List<String> frenchWords, MyIntegerDoubleHashMap sentenceTotalE, List<Integer> frIdxBwdList, List<Integer> enIdxBwdList, List<Float> tE_over_F_BwdList) {
        //====================== BACKWARD t(e|f) ======================
        frenchWords.add(0, "null"); //add NULL token
        boolean isFrBwdListAdded = true;
        for (String en : englishWords) {
            int enIdx = enIndexer.indexOf(en.toLowerCase());
            enIdxBwdList.add(enIdx);
            for (String fr : frenchWords) {
                int frIdx = frIndexer.indexOf(fr.toLowerCase());
                if (isFrBwdListAdded) frIdxBwdList.add(frIdx); // add 0 at first

                float t_E_over_F = 0;
                if (iter == 0) {
//                    if (fr.equals("null"))
//                        t_E_over_F += .2;
//                    else
//                        t_E_over_F += .8 / enIndexer.size();//
                    t_E_over_F += 1. / enIndexer.size();
                    probEoverF[enIdx][frIdx] = t_E_over_F;
                } else
                    t_E_over_F = probEoverF[enIdx][frIdx];

                tE_over_F_BwdList.add(t_E_over_F); // add t(F|NULL) for all F at first
                sentenceTotalE.increment(enIdx, t_E_over_F);
            }
            isFrBwdListAdded = false; // next loop will not redundantly add into enIdxList
        }
        frenchWords.remove(0); // remove token NULL
    }

    private void calculatePartialCountsBackward(float[][] countEoverF, float[] totalF, MyIntegerDoubleHashMap sentenceTotalE, List<Integer> frIdxBwdList, List<Integer> enIdxBwdList, List<Float> tE_over_F_BwdList) {
        // collect partial counts
        int c = 0;
        for (int enIdx : enIdxBwdList) {
            for (int frIdx : frIdxBwdList) {
                float t_E_over_F = tE_over_F_BwdList.get(c);
                double s_total_E = sentenceTotalE.get(enIdx);
                countEoverF[enIdx][frIdx] += t_E_over_F / s_total_E;
                totalF[frIdx] += t_E_over_F / s_total_E;
                c++;
            }
        }
    }

    private double updateProbabilitiesAndLossBackward(float[][] countEoverF, float[] totalF) {
        double lossBW;// ==================MASTER UPDATING ====================
        System.out.println("Reupdating t(e|f)");
        lossBW = 0.0;
        for (int e = 0; e < enIndexer.size(); e++) {
            float[] currentProbEoverFMap = probEoverF[e];
            float[] currentCountEoverFMap = countEoverF[e];
            for (int f = 0; f < frIndexer.size(); f++) {
                float newVal = currentCountEoverFMap[f] / totalF[f];
                lossBW += Math.abs(currentProbEoverFMap[f] - newVal);
                currentProbEoverFMap[f] = newVal;
            }
        }
        lossBW /= (enIndexer.size() * frIndexer.size());
        return lossBW;
    }


    // ============ ALIGNMENT SHOWTIME ==============
    @Override
    public Alignment alignSentencePair(SentencePair sentencePair) {
        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();

        Alignment forwardAlignment = getForwardAlignments(englishWords, frenchWords);
        Alignment backwardAlignment = getBackwardAlignments(englishWords, frenchWords);

        forwardAlignment.getSureAlignments().retainAll(backwardAlignment.getSureAlignments());
        return forwardAlignment;
    }

    private Alignment getForwardAlignments(List<String> englishWords, List<String> frenchWords) {
        // FORWARD
        Alignment alignment = new Alignment();
        for (int j = 0; j < frenchWords.size(); j++) {
            int frIdx = frIndexer.indexOf(frenchWords.get(j).toLowerCase());
            float maxProb = probFoverE[frIdx][0];
            int bestPosition = -1; // NULL position
            for (int i = 0; i < englishWords.size(); i++) {
                int enIdx = enIndexer.indexOf(englishWords.get(i).toLowerCase());
                float t_F_over_E = probFoverE[frIdx][enIdx];

                if (maxProb < t_F_over_E) {
                    maxProb = t_F_over_E;
                    bestPosition = i; // we already count NULL position = 0
                }
            }
            // align
            alignment.addAlignment(bestPosition, j, true); // if add NULL to english sentence then (--bestPosition)
        }
        return alignment;
    }

    private Alignment getBackwardAlignments(List<String> englishWords, List<String> frenchWords) {
        // BACKWARD
        Alignment alignment = new Alignment();
        for (int i = 0; i < englishWords.size(); i++) {
            int enIdx = enIndexer.indexOf(englishWords.get(i).toLowerCase());
            float maxProb = probEoverF[enIdx][0];
            int bestPosition = -1; // NULL position
            for (int j = 0; j < frenchWords.size(); j++) {
                int frIdx = frIndexer.indexOf(frenchWords.get(j).toLowerCase());
                float t_E_over_F = probEoverF[enIdx][frIdx];

                if (maxProb < t_E_over_F) {
                    maxProb = t_E_over_F;
                    bestPosition = j; // we already count NULL position = 0
                }
            }
            alignment.addAlignment(i, bestPosition, true); // if add NULL to english sentence then (--bestPosition)
        }
        return alignment;
    }


    // ==================== DEBUG ===============================
    private void debugIntIntOpenHashMap(IntIntOpenHashMap map) {
        System.out.println("\nContent of IntIntOpenHashMap with enIndexer");
        for (IntIntOpenHashMap.Entry entry : map.entrySet()) {
            System.out.format("Key=%s\tVal=%d\n", enIndexer.get(entry.getKey()), entry.getValue());
        }
        System.out.println("-----------------");
    }

    private void debugIntDoubleOpenHashMap(IntDoubleOpenHashMap map) {
        System.out.println("\nContent of IntDoubleOpenHashMap with frIndexer");
        for (IntDoubleOpenHashMap.Entry entry : map.entrySet()) {
            System.out.format("Key=%s\tVal=%.4f\n", frIndexer.get(entry.getKey()), entry.getValue());
        }
        System.out.println("-----------------");
    }

    private void debugLongDoubleOpenHashMap(LongDoubleOpenHashMap map) {
        System.out.println("\nContent of IntDoubleOpenHashMap with frIndexer");
        for (LongDoubleOpenHashMap.Entry entry : map.entrySet()) {
            long key = entry.getKey();
            int[] decoded = BitPackingUtility.bigramBitPackingDecode(key);
            System.out.format("KeyPair= (%s + %s)\tVal=%d\n", frIndexer.get(decoded[0]), enIndexer.get(decoded[1]), entry.getValue());
        }
        System.out.println("-----------------");
    }

    private void debugBigramPackingLongIntOpenHashMap(LongIntOpenHashMap map) {
        System.out.println("\nContent of LongIntOpenHashMap");
        for (LongIntOpenHashMap.Entry entry : map.entrySet()) {
            long key = entry.getKey();
            int[] decoded = BitPackingUtility.bigramBitPackingDecode(key);
            System.out.format("KeyPair= (%s + %s)\tVal=%d\n", frIndexer.get(decoded[0]), enIndexer.get(decoded[1]), entry.getValue());
        }
        System.out.println("-----------------");
    }

    private void debugBigramPackingLongDoubleOpenHashMap(LongDoubleOpenHashMap map) {
        System.out.println("\nContent of Long Double OpenHashMap");
        for (LongDoubleOpenHashMap.Entry entry : map.entrySet()) {
            long key = entry.getKey();
            int[] decoded = BitPackingUtility.bigramBitPackingDecode(key);
            System.out.format("KeyPair=%d + %d (%s + %s)\tVal=%.4f\n", decoded[0], decoded[1], frIndexer.get(decoded[0]), enIndexer.get(decoded[1]), entry.getValue());
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
