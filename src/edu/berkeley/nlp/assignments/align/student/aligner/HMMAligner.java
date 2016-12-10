package edu.berkeley.nlp.assignments.align.student.aligner;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.*;

/**
 * Created by Gorilla on 12/6/2016.
 */
public class HMMAligner implements WordAligner {
    private static StringIndexer enIndexer;
    private static StringIndexer frIndexer;
    double[][] probEoverF; //forward
    double[][] probFoverE; //backward

    public HMMAligner(Iterable<SentencePair> trainingData) {
        HMMForwardAligner fwdAligner = new HMMForwardAligner(trainingData);
        probEoverF = fwdAligner.getProbEoverF();
        enIndexer = fwdAligner.getEnIndexer();
        frIndexer = fwdAligner.getFrIndexer();
        fwdAligner = null;

        HMMBackwardAligner bwdAligner = new HMMBackwardAligner(trainingData);
        probFoverE = bwdAligner.getProbFoverE();
        bwdAligner = null;
    }

    @Override
    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignmentForward = new Alignment();

        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();
        int[] enIdx = new int[englishWords.size()];
        for (int j = 0; j < englishWords.size(); j++) {
            enIdx[j] = enIndexer.indexOf(englishWords.get(j).toLowerCase());
        }

        int[] frIdx = new int[frenchWords.size()];
        for (int i = 0; i < frenchWords.size(); i++) {
            frIdx[i] = frIndexer.indexOf(frenchWords.get(i).toLowerCase());
        }

        // init sentence-wise variables
        double[] prior = initPriorLog(englishWords.size());
        double[][] transition = initTransitionMatrix(englishWords.size());

        List<Integer> fwdDecoded = viterbiDecodeForward(enIdx, frIdx, prior, transition);
//        HashMap<Integer, Integer> forwardMap = new HashMap<>();
        for (int i = 0; i < frIdx.length; i++) {
            alignmentForward.addAlignment(fwdDecoded.get(i), i, true);
//            forwardMap.put(fwdDecoded.get(i), i);
        }

        Alignment alignmentBackward = new Alignment();
        prior = initPriorLog(frenchWords.size());
        transition = initTransitionMatrix(frenchWords.size());
        List<Integer> bwdDecoded = viterbiDecodeBackward(frIdx, enIdx, prior, transition);
//        HashMap<Integer, Integer> backwardMap = new HashMap<>();
        for (int i = 0; i < enIdx.length; i++) {
            alignmentBackward.addAlignment(i, bwdDecoded.get(i), true);
//            backwardMap.put(i, bwdDecoded.get(i));
        }

//        Set<Integer> forwardKeySet = forwardMap.keySet();
//        Set<Integer> backwardKeySet = backwardMap.keySet();
//
//        for (int e : forwardKeySet) {
//            int forwardVal = forwardMap.get(e);
//            if (backwardKeySet.contains(e)) {
//                int backwardVal = backwardMap.get(e);
//                if (backwardVal == forwardVal)
//                    alignment.addAlignment(e, backwardVal, true);
//            }
//        }
        alignmentForward.getSureAlignments().retainAll(alignmentBackward.getSureAlignments());

        return alignmentForward;
    }

    private List<Integer> viterbiDecodeForward(int[] rowIdx, int[] colIdx, double[] prior, double[][] transition) {
        int E = rowIdx.length;
        int F = colIdx.length;
        double[][] B = new double[F][E]; // N x T // back pointer
        List<List<List<Integer>>> V = new ArrayList<>(); // T x N x (colIdx.length)

        // base case for B
        for (int i = 0; i < E; i++) {
            B[0][i] = prior[i] + probEoverF[i][0];
        }
        // base case for V
        for (int i = 0; i < F; i++) {
            V.add(i, new ArrayList<>());
            for (int j = 0; j < E; j++) {
                V.get(i).add(j, new ArrayList<>());
            }
        }
        // init V with init EN states
        for (int j = 0; j < E; j++) {
//            V.get(0).get(j).add(rowIdx[j]);
            V.get(0).get(j).add(j);

        }
//        System.out.println("After init the content of V = " + V);
//        print2DArray(B, "B");
        // decode
        for (int t = 1; t < F; t++) { // from 1 -> T
            for (int i = 0; i < E; i++) {
                int maxIdx = -1;
                double maxVal = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < E; k++) {
//                    double current = B[t-1][k] + transition[k][i] + probEoverF[rowIdx[i]][colIdx[t]];
                    double current = B[t-1][k] + transition[i][k] + probEoverF[rowIdx[i]][colIdx[t]];
                    if (maxVal < current) {
                        maxVal = current;
                        maxIdx = k;
                    }
                }
                // update
                B[t][i] = maxVal;
//                System.out.println("MaxVal = " + maxVal + " maxIDx = " + maxIdx);
                List<Integer> currentList = new ArrayList<>(V.get(t-1).get(maxIdx)); // copy, not reference
//                System.out.println("Current List at (t-1)= " + (t-1) + " and i = " + i + " is " + currentList);
//                currentList.add(rowIdx[i]);
                currentList.add(i);
                V.get(t).add(i, currentList);
            }
//            System.out.println("After step " + t + " the content of V = " + V);
        }

        // now backtracking
        List<Integer> maxDecoded = new ArrayList<>();
        for (int i = F-1; i >= 0; i--) {
            maxDecoded.add(0, argmax(B, i));
        }
        return maxDecoded;
    }

    private List<Integer> viterbiDecodeBackward(int[] rowIdx, int[] colIdx, double[] prior, double[][] transition) {
        int F = rowIdx.length;
        int E = colIdx.length;
        double[][] B = new double[E][F]; // N x T // back pointer
        List<List<List<Integer>>> V = new ArrayList<>(); // T x N x (colIdx.length)

        // base case for B
        for (int i = 0; i < F; i++) {
            B[0][i] = prior[i] + probFoverE[i][0];
        }
        // base case for V
        for (int i = 0; i < E; i++) {
            V.add(i, new ArrayList<>());
            for (int j = 0; j < F; j++) {
                V.get(i).add(j, new ArrayList<>());
            }
        }
        // init V with init EN states
        for (int j = 0; j < F; j++) {
//            V.get(0).get(j).add(rowIdx[j]);
            V.get(0).get(j).add(j);

        }
//        System.out.println("After init the content of V = " + V);
//        print2DArray(B, "B");
        // decode
        for (int t = 1; t < E; t++) { // from 1 -> T
            for (int i = 0; i < F; i++) {
                int maxIdx = -1;
                double maxVal = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < F; k++) {
//                    double current = B[t-1][k] + transition[k][i] + probFoverE[rowIdx[i]][colIdx[t]];
                    double current = B[t-1][k] + transition[i][k] + probFoverE[rowIdx[i]][colIdx[t]];
                    if (maxVal < current) {
                        maxVal = current;
                        maxIdx = k;
                    }
                }
                // update
                B[t][i] = maxVal;
//                System.out.println("MaxVal = " + maxVal + " maxIDx = " + maxIdx);
                List<Integer> currentList = new ArrayList<>(V.get(t-1).get(maxIdx)); // copy, not reference
//                System.out.println("Current List at (t-1)= " + (t-1) + " and i = " + i + " is " + currentList);
//                currentList.add(rowIdx[i]);
                currentList.add(i);
                V.get(t).add(i, currentList);
            }
//            System.out.println("After step " + t + " the content of V = " + V);
        }

        // now backtracking
        List<Integer> maxDecoded = new ArrayList<>();
        for (int i = E-1; i >= 0; i--) {
            maxDecoded.add(0, argmax(B, i));
        }


//        print2DArray(B, "backpointer");
//        System.out.println(V);

        return maxDecoded;
    }

    /**
     * helper for viterbi, dealing with back pointer matrix with size T x N
     * Usage: Back track for each time step from t = 0 => t = T-1
     */
    private int argmax(double[][] backPtr, int rowIdx) {
        int maxIdx = 0;
        double maxVal = Double.NEGATIVE_INFINITY;
        for (int col = 0; col < backPtr[rowIdx].length; col++) {
            if (maxVal < backPtr[rowIdx][col]) {
                maxVal = backPtr[rowIdx][col];
                maxIdx = col;
            }
        }
        return maxIdx;
    }

    private double[] initPriorLog(int size) {
        double[] P = new double[size];
        Arrays.fill(P, Math.log(1 / (double) size));
        return P;
    }


    private double[][] initTransitionMatrix(int size) {
        return laplaceTransitionInitLog(size);
    }

    private double[][] laplaceTransitionInitLog(int size) {
//        System.out.println("Using Laplacian distribution for transition matrix");
        double[][] T = new double[size][size];

        for (int row = 0; row < size; row ++) {
            for (int col = 0; col < size; col++) {
                T[row][col] = laplacianSample((double)(row-col), 1., 5.);
            }
        }

        for (int col=0; col < size; col++) {
            double colSum = 0;
            for (int row = 0; row < size; row++) {
                colSum += T[row][col];
            }
            for (int row = 0; row < size; row++) {
                T[row][col] /= colSum;
            }
        }
//        for (int r=0; r < size; r++) {
//            double rSum = 0;
//            for (int c = 0; c < size; c++) {
//                rSum += T[r][c];
//            }
//            for (int c = 0; c < size; c++) {
//                T[r][c] /= rSum;
//            }
//        }

        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                T[i][j] = Math.log(T[i][j]);

        return T;
    }

    private double laplacianSample(double x, double mu, double b) {
        return ( Math.exp( - Math.abs(x - mu) / b) ) /  (2.0 * b);
    }

}
