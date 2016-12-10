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
public class IBM1ForwardAligner implements WordAligner {

    private static StringIndexer enIndexer = new StringIndexer();
    private static StringIndexer frIndexer = new StringIndexer();
    private static float[][] probFoverE;

    // CONSTRUCTOR
    public IBM1ForwardAligner(Iterable<SentencePair> trainingData, int targetNumIterations) {
        System.out.println("Training for IBM Model 1");
        initializeEM(trainingData);

        // convergence criteria
        double delta = 1e-7;
        forwardModel1(trainingData, targetNumIterations, delta);
    }

    public IBM1ForwardAligner(Iterable<SentencePair> trainingData) {
        System.out.println("Training for IBM Model 1");
        initializeEM(trainingData);

        // convergence criteria
        int targetNumIterations = 50;
        double delta = 1e-7;
        forwardModel1(trainingData, targetNumIterations, delta);
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
        probFoverE = new float[frVocabSize][enVocabSize];
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


    // ============ ALIGNMENT SHOWTIME ==============
    @Override
    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignment = new Alignment();
        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();

        alignment = getForwardAlignments(englishWords, frenchWords, alignment);
        return alignment;
    }

    private Alignment getForwardAlignments(List<String> englishWords, List<String> frenchWords, Alignment alignment) {
        // FORWARD
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

    public static float[][] getProbFoverE() {
        return probFoverE;
    }

    public static StringIndexer getEnIndexer() {
        return enIndexer;
    }

    public static StringIndexer getFrIndexer() {
        return frIndexer;
    }
}
