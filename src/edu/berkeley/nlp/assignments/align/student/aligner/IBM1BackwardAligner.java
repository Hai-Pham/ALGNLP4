package edu.berkeley.nlp.assignments.align.student.aligner;

import edu.berkeley.nlp.assignments.align.student.util.*;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.*;

/**
 * Created by Gorilla on 12/3/2016.
 * model t(e|f)
 */
public class IBM1BackwardAligner implements WordAligner {

    private static StringIndexer enIndexer = new StringIndexer();
    private static StringIndexer frIndexer = new StringIndexer();
    private static float[][] probEoverF;

    // CONSTRUCTOR
    public IBM1BackwardAligner(Iterable<SentencePair> trainingData, int targetNumIterations) {
        System.out.println("Training for IBM Model 1");
        initializeEM(trainingData);

        // convergence criteria
        double delta = 1e-7;
        backwardModel1(trainingData, targetNumIterations, delta);
    }

    public IBM1BackwardAligner(Iterable<SentencePair> trainingData) {
        System.out.println("Training for IBM Model 1");
        initializeEM(trainingData);

        // convergence criteria
        int targetNumIterations = 50;
        double delta = 1e-7;
        backwardModel1(trainingData, targetNumIterations, delta);
    }

    // ============ INITIALIZATION ==============
    private void initializeEM(Iterable<SentencePair> trainingData) {
        // initialization
        System.out.println("Scanning through all data and initializing EM...");
        System.out.println("Index of NULL token is " + enIndexer.addAndGetIndex("null")); // NULL TOKEN for ENGLISH
        frIndexer.add("null");

        int pairCount = 0;
        int maxFrenchLen = 0;
        int maxEnglishLen = 0;
        for (SentencePair pair : trainingData) {
            pairCount++;
            if (pairCount % 1000 == 0) System.out.println(pairCount + " sentences loaded");
            List<String> englishWords = pair.getEnglishWords();
            List<String> frenchWords = pair.getFrenchWords();
            if (maxEnglishLen < englishWords.size()) maxEnglishLen = englishWords.size();
            if (maxFrenchLen < frenchWords.size()) maxFrenchLen = frenchWords.size();

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
        System.out.println("Max ENG sentence length is " + maxEnglishLen + " and for FR is " + maxFrenchLen);
        System.out.println("Initializing done!");

        // init 2D arrays
        probEoverF = new float[enVocabSize][frVocabSize];
    }

    // ============ BACKWARD MODEL1 ==============
    private void backwardModel1(Iterable<SentencePair> trainingData, int targetNumIterations, double delta) {
        int iter;// ==================BACKWARD================================================
        iter = 0;
        double lossBW = 1;
        while ((iter < targetNumIterations) && (lossBW > delta)) {
            System.out.println("\n-----------------\nIBM1 MODEL BACKWARD Iteration " + iter + "\n-----------------\n");
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
        Alignment alignment = new Alignment();
        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();

        alignment = getBackwardAlignments(englishWords, frenchWords, alignment);
        return alignment;
    }


    private Alignment getBackwardAlignments(List<String> englishWords, List<String> frenchWords, Alignment alignment) {
        // BACKWARD
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
            // align
            alignment.addAlignment(i, bestPosition, true); // if add NULL to english sentence then (--bestPosition)
        }
        return alignment;
    }

    public static float[][] getProbEoverF() {
        return probEoverF;
    }

    public static StringIndexer getEnIndexer() {
        return enIndexer;
    }

    public static StringIndexer getFrIndexer() {
        return frIndexer;
    }
}
