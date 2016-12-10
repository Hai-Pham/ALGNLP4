package edu.berkeley.nlp.assignments.align.student.aligner;

import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.StringIndexer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Gorilla on 12/6/2016.
 * Model t(f|e)
 */
public class HMMForwardAligner implements WordAligner {

    private static StringIndexer enIndexer = new StringIndexer();
    private static StringIndexer frIndexer = new StringIndexer();
    private static int maxFrenchLen = 0;
    private static int maxEnglishLen = 0;

    private static double[][] probEoverF; // E is state, F is word (size is N x T)

    public HMMForwardAligner(Iterable<SentencePair> trainingData) {
        initializeHMM(trainingData);
        trainFwdHMM(trainingData);
    }

    private void initializeHMM(Iterable<SentencePair> trainingData) {
        initIndexer(trainingData);

        probEoverF = new double[enIndexer.size()][frIndexer.size()];
        for (double[] row : probEoverF)
            Arrays.fill(row, Math.log(1 / (double) frIndexer.size()));


        System.out.println("Shape of EMISSION matrix is (" + probEoverF.length + "," + probEoverF[0].length + ")");
    }

    private void alphaBetaTest(Iterable<SentencePair> trainingData) {
        for (SentencePair pair : trainingData) {
            List<String> englishWords = pair.getEnglishWords();
            List<String> frenchWords = pair.getFrenchWords();

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

            double[][] alpha = calculateAlphaLog(enIdx, frIdx, prior, transition);
            double[][] beta = calculateBetaLog(enIdx, frIdx, transition);
//            double[][] gamma = calculateGammaLog(alpha, beta);

//            assert (alpha.length == beta.length);
//            print2DArray(transition, "transition");
            print2DArray(alpha, "alpha");
            print2DArray(beta, "beta");
//            print2DArray(gamma, "gamma");

//            //log version
//            double[] sumA = new double[alpha.length];
//            for (int i = 0; i < alpha.length; i++)
//                sumA[i] = alpha[i][alpha[0].length - 1];
//            double a = SloppyMath.logAdd(sumA);
//
//            //log version
//            double[] sumB = new double[beta.length];
//            for (int i = 0; i < beta.length; i++)
//                sumB[i] = prior[i] + probEoverF[i][frIdx[0]] + beta[i][0];
//            double b = SloppyMath.logAdd(sumB);

            // log version 2
            double[] sumA = new double[alpha.length];
            for (int i = 0; i < alpha[0].length; i++)
                sumA[i] = alpha[alpha.length - 1][i];
            double a = SloppyMath.logAdd(sumA);

            //log version
            double[] sumB = new double[beta[0].length];
            for (int i = 0; i < beta[0].length; i++)
                sumB[i] = prior[i] + probEoverF[i][frIdx[0]] + beta[0][i];
            double b = SloppyMath.logAdd(sumB);

            System.out.println("a = " + a + " b = " + b);
        }
    }

    private void initIndexer(Iterable<SentencePair> trainingData) {
        int pairCount = 0;
        for (SentencePair pair : trainingData) {
            if (pairCount % 1000 == 0) System.out.println(pairCount+ " sentences loaded");
            pairCount++;

            List<String> englishWords = pair.getEnglishWords();
            List<String> frenchWords = pair.getFrenchWords();
            if (maxEnglishLen < englishWords.size()) maxEnglishLen = englishWords.size();
            if (maxFrenchLen < frenchWords.size()) maxFrenchLen = frenchWords.size();

            // get english words
            for (String en : englishWords) enIndexer.add(en.toLowerCase());
            // and scan thru french words
            for (String fr : frenchWords) frIndexer.add(fr.toLowerCase());
        }
        int frVocabSize = frIndexer.size();
        int enVocabSize = enIndexer.size();
        System.out.println("Size of FR vocab is " + frVocabSize + " size of EN vocab is " + enVocabSize);
        System.out.println("Max ENG sentence length is " + maxEnglishLen + " and for FR is " + maxFrenchLen);
        System.out.println("Initializing of Indexer is done!");
    }

    private double[] initPriorLog(int size) {
        double[] P = new double[size];
        Arrays.fill(P, Math.log(1 / (double) size));
        return P;
    }

    private double[][] initTransitionMatrix(int size) {
//        return uniformTransitionInit(size);
        return laplaceTransitionInitLog(size);
    }

    private double[][] uniformTransitionInit(int size) {
//        System.out.println("Using Uniform distribution for transition matrix");
        double[][] T = new double[size][size];
        for (double[] row : T) {
            Arrays.fill(row, Math.log(1 / (double) size));
        }
        return T;
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

        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                T[i][j] = Math.log(T[i][j]);

        return T;
    }

    private double laplacianSample(double x, double mu, double b) {
        return ( Math.exp( - Math.abs(x - mu) / b) ) /  (2.0 * b);
    }

    // ======================= FORWARD BACKWARD =====================
    private void trainFwdHMM(Iterable<SentencePair> trainingData) {
        // convergence criteria
        int targetNumIterations = 10;
        double delta = 1e-5;

        int iter = 0;
        double lossFW = 1.;
        while ((iter < targetNumIterations) && (lossFW > delta)) {
            System.out.println("\n---------------------\n" + "HMM FORWARD Iteration" + iter + "\n--------------------\n");
            double[][] countEoverF = new double[enIndexer.size()][frIndexer.size()];
            double[] totalE = new double[enIndexer.size()];
            for (double[] row: countEoverF)
                Arrays.fill(row, Double.NEGATIVE_INFINITY);
            Arrays.fill(totalE, Double.NEGATIVE_INFINITY);

//            double[] totalPrior = new double[enIndexer.size()];
//            double priorNormalize = 0;

            int count = 0;
            for (SentencePair pair : trainingData) {
                if (count % 1000 == 0)
                    System.out.println(count + " pairs processed");
                count++;

                List<String> englishWords = pair.getEnglishWords();
                List<String> frenchWords = pair.getFrenchWords();
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

                // E-step
                // GAMMA
                double[][] alpha = calculateAlphaLog(enIdx, frIdx, prior, transition);
                double[][] beta = calculateBetaLog(enIdx, frIdx, transition);

//                print2DArray(alpha, "alpha");
//                print2DArray(beta, "beta");
//                print2DArray(transition, "transition");

                // TODO: compute Xi

                double[] sentenceTotalF = new double[frIdx.length];
                Arrays.fill(sentenceTotalF, Double.NEGATIVE_INFINITY);
                for (int i = 0; i < frIdx.length; i++) {
                    for (int j = 0; j < enIdx.length; j++) {
                        sentenceTotalF[i] = SloppyMath.logAdd(alpha[i][j] + beta[i][j], sentenceTotalF[i]);
                    }
                }
                // accumulate partial counts for emission
                for (int i = 0; i < frIdx.length; i++) {
                    for (int j = 0; j < enIdx.length; j++) {
                        double gamma = alpha[i][j] + beta[i][j] - sentenceTotalF[i];
                        countEoverF[enIdx[j]][frIdx[i]] = SloppyMath.logAdd(countEoverF[enIdx[j]][frIdx[i]], gamma);
                        totalE[enIdx[j]] = SloppyMath.logAdd(totalE[enIdx[j]], gamma);
                    }
                }

                        // accumulate partia counts for priors
                // TODO: update this
//                for (int i = 0; i < enIdx.length; i++) { // row of gamma
//                    totalPrior[enIdx[i]] += gamma[i][0];
//                    priorNormalize += gamma[i][0];
//                }

            } // end for all sentence pairs

            // now reupdating master t(e|f)
            System.out.println("Reupdaing table t(e|f)");
//            lossFW = 0.;
            for (int e = 0; e < enIndexer.size(); e++) {
                for (int f = 0; f< frIndexer.size(); f++) {
                    double oldValue = probEoverF[e][f];
//                    lossFW += Math.abs(Math.exp(oldValue) - Math.exp(countEoverF[e][f] - totalE[e]));
                    probEoverF[e][f] = countEoverF[e][f] - totalE[e];
                }
            }
//            lossFW /= (enIndexer.size() * frIndexer.size());
//            System.out.println("Loss = " + lossFW);

            // now updating prior
//            for (int e = 0; e < enIndexer.size(); e++) {
//                fwdPrior[e] = (double) (totalPrior[e] / priorNormalize);
//            }

//            print2DArray(probEoverF, "t(f|e)");

            // to the next loop if not converged
            countEoverF = null;
            totalE = null;
            iter++;
        } // end while
    }

    /**
     * Calculate alpha (forward) in log space
     * TODO: Right now NULL case is not handled
     * Usage:
     * Extract emissions from t(f|e) table, while reuse global prior and transition
     *
     * @param colIdx: given a sentence, extract the label index of the sentence words
     */
    private double[][] calculateAlphaLog(int[] rowIdx, int[] colIdx, double[] prior, double[][] transition) {
        int E = rowIdx.length; // eng
        int F = colIdx.length; // fr
        double[][] alpha = new double[F][E];

        //base case
        for (int i = 0; i < E; i++) {
            alpha[0][i] = prior[i] + probEoverF[rowIdx[i]][colIdx[0]];
//            System.out.println(alpha[0][i] + " " + prior[i] + " " + probEoverF[rowIdx[i]][colIdx[0]]);
        }

        // forward
        for (int t = 1; t < F; t++) {
            for (int i = 0; i < E; i++) {
                double logSum = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < E; k++) {
                    logSum = SloppyMath.logAdd(logSum, alpha[t - 1][k] + transition[i][k]);
                }
                // calculate next timestep alpha
                alpha[t][i] = probEoverF[rowIdx[i]][colIdx[t]] + logSum;
            }

        }
        return alpha;
    }

    /**
     * Calculate beta (backward) in log space
     *
     * @param colIdx: given a sentence, extract the label index of the sentence words
     * @return
     */
    private double[][] calculateBetaLog(int[] rowIdx, int[] colIdx, double[][] transition) {
        int N = rowIdx.length;
        int T = colIdx.length;
        double[][] beta = new double[T][N]; // N x T

        // base case, no need to normalize
        for (int i = 0; i < N; i++) {
            beta[T - 1][i] = 0; // log(1) = 0
        }

        // backward
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < N; i++) {
                double logSum = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < N; k++) {
                    logSum = SloppyMath.logAdd(logSum, beta[t + 1][k] + transition[k][i] + probEoverF[rowIdx[k]][colIdx[t + 1]]);
                }
                beta[t][i] = logSum;
            }
        }
        return beta;
    }

    /**
     * Given alpha and beta, calculate gamma which has the exact dimension
     */
    private double[][] calculateGammaLog(double[][] alpha, double[][] beta) {
        int N = alpha.length;
        int T = alpha[0].length;
        double[][] gamma = new double[N][T]; // N x T
        for (double[] row: gamma)
            Arrays.fill(row, Double.NEGATIVE_INFINITY);

        for (int t = 0; t < T; t++) {
            // calculate normalization factor first
            double[] normalize = new double[N];
            for (int i = 0; i < N; i++) {
                normalize[i] = (alpha[i][t] + beta[i][t]);
            }
            double norm = SloppyMath.logAdd(normalize);
            // now update gamma
            for (int i = 0; i < N; i++) {
                gamma[i][t] = (alpha[i][t] + beta[i][t] - norm);
            }
        }
        return gamma;
    }

    /**
     * Viterbi decoder using for alignment
     * Given FR decode EN
     */
    private List<Integer> viterbiDecode(int[] rowIdx, int[] colIdx, double[] prior, double[][] transition) {
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


    private double[][][] calculateXi(double[][] alpha, double[][] beta) {
        int N = alpha.length;
        int T = alpha[0].length;
        double[][][] xi = new double[N][N][T];

        for (int t = 0; t < T; t++) {
            // calculate normalization, alpha & beta in logspace, transition and emission need a conversion
        }
        return xi;
    }

    // ======================= ALIGN NOW =============================
    @Override
    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignment = new Alignment();
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

        List<Integer> decoded = viterbiDecode(enIdx, frIdx, prior, transition);
        System.out.println(print1DArray(enIdx));
        System.out.println(print1DArray(frIdx));
        System.out.println(decoded);
        for (int i = 0; i < frIdx.length; i++) {
            alignment.addAlignment(decoded.get(i), i, true);
        }

        return alignment;
    }

    // ======================= DEBUG ==================================
    public void print3DArray(double[][][] a, String label) {
        System.out.println("Debugging content of " + label + " [\n------");
        for (double[][] row : a) {
            for (double[] col : row)
                for (double f : col)
                    System.out.print(f + ", ");
            System.out.println();
            System.out.println("------");
        }
        System.out.println("]");
    }

    public void print2DArray(double[][] a, String label) {
        System.out.println("Debugging content of " + label + " [");
        for (double[] row : a) {
            for (double f : row)
                System.out.print(f + ", ");
            System.out.println();
        }
        System.out.println("]");
    }

    public void debugTransitionMatrix(double[][] a) {
        System.out.println("Debugging transition matrix with length  " + a.length);
        for (int row = 0; row < a.length; row++) {
            for (int col = 0; col < a.length; col++) {
                // for a(i)(j) sum all k of a(k)(j+1) should equal to 1.0
                double sum = 0;
                for (int k = 0; k < a.length; k++) {
                    sum += a[k][col];
                }
                assert (sum == 1);
//                System.out.print(sum + " ");
            }
        }
        System.out.println("Debug is done successfully");
    }

    public String print1DArray(int[] a) {
        String s = "";
        for (int i: a)
            s += i + ",";
        return s;
    }

    public static StringIndexer getEnIndexer() {
        return enIndexer;
    }

    public static StringIndexer getFrIndexer() {
        return frIndexer;
    }

    public static double[][] getProbEoverF() {
        return probEoverF;
    }
}
