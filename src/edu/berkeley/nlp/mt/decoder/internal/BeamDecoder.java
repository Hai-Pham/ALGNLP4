package edu.berkeley.nlp.mt.decoder.internal;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.MurmurHash;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.PhraseTableForSentence;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;
import edu.berkeley.nlp.util.BoundedList;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.IntPriorityQueue;
import edu.berkeley.nlp.util.StringIndexer;

public class BeamDecoder implements Decoder
{

	private final StringIndexer lexIndexer;

	private final NgramLanguageModel lm;

	private PhraseTable tm;

	public BeamDecoder(NgramLanguageModel lm, PhraseTable tm, StringIndexer wordIndexer) {
		this.lm = lm;
		this.tm = tm;
		this.lexIndexer = wordIndexer;

	}

	public List<ScoredPhrasePairForSentence> decode(List<String> sentence) {
		int[] intSentence = new int[sentence.size()];
		for (int i = 0; i < sentence.size(); ++i) {
			intSentence[i] = lexIndexer.indexOf(sentence.get(i));
		}
		final int lmOrder = lm.getOrder();
		int length = sentence.size();
		final int k = 1000;
		final boolean hash = false;
		final int arraySize = (int) (k * 1.2) + 1;
		final double[][] scoreChart = new double[length + 1][arraySize];
		final int[][][] lmContexts = new int[length + 1][arraySize][lmOrder - 1];
		final int[][] lmContextLengths = new int[length + 1][arraySize];
		final int[][] transBackpointers = new int[length + 1][arraySize];
		final int[][] prevStateBackpointers = new int[length + 1][arraySize];
		final int[][] startBackpointers = new int[length + 1][arraySize];
		fill(transBackpointers, -1);
		fill(lmContextLengths, -1);
		fill(prevStateBackpointers, -1);
		fill(startBackpointers, -1);

		// note: these beams are meant to keep the k *best* (highest scoring) items.
		// However, this beam is a min-heap because we would like to be able to delete the lowest scoring item whenever we add a new item (and the beam is full). 
		final IntPriorityQueue[] beams = new IntPriorityQueue[length + 1];
		for (int i = 0; i < length + 1; ++i) {
			if (!hash) beams[i] = new IntPriorityQueue(arraySize, arraySize);
		}
		fill(scoreChart, Double.NEGATIVE_INFINITY);
		PhraseTableForSentence tmState = tm.initialize(sentence);
		for (int start = 0; start < length; ++start) {
			final boolean isBegin = start == 0;
			for (int end = start + 1; end <= start + tmState.getMaxPhraseLength(); ++end) {
				final List<ScoredPhrasePairForSentence> translations = tmState.getScoreSortedTranslationsForSpan(start, end);
				if (translations == null) continue;
				final int prevStateEnd = isBegin ? 1 : arraySize;
				for (int prevState = 0; prevState < prevStateEnd; ++prevState) {
					double prevScore = isBegin ? 0.0 : scoreChart[start][prevState];
					if (prevScore == Double.NEGATIVE_INFINITY) continue;
					int[] prevLmState = isBegin ? new int[] { lexIndexer.indexOf(NgramLanguageModel.START) } : lmContexts[start][prevState];
					final int prevLmStateLength = isBegin ? 1 : lmContextLengths[start][prevState];
					int[] lmStateBuf = CollectionUtils.copyOf(prevLmState, prevLmStateLength + tmState.getMaxPhraseLength() + 1);
					final int numTranslations = translations.size();
					for (int transIndex = 0; transIndex < numTranslations; ++transIndex) {
						innerLoop(lmOrder, length, k, hash, scoreChart, lmContexts, lmContextLengths, transBackpointers, prevStateBackpointers,
							startBackpointers, beams, start, translations, prevState, prevScore, prevLmStateLength, lmStateBuf, transIndex, lexIndexer, lm);
					}
				}
			}
		}

		int startPos = length;
		double max = Double.NEGATIVE_INFINITY;
		int stateIndex = -1;
		for (int index = 0; index < arraySize; ++index) {
			double score = scoreChart[startPos][index];
			if (score > max) {
				stateIndex = index;
				max = score;
			}
		}
		if (length > 0 && stateIndex == -1) {
		  throw new RuntimeException("Error in decoder: Language model probably returned NEGATIVE_INFINITY or NaN");
		}
		List<ScoredPhrasePairForSentence> translation = new ArrayList<ScoredPhrasePairForSentence>();
		double score = 0.0;
		while (startPos > 0) {
			int prevStartPos = startBackpointers[startPos][stateIndex];
			List<ScoredPhrasePairForSentence> lexSortedTranslaitons = tmState.getScoreSortedTranslationsForSpan(prevStartPos, startPos);
			final int transIndex = transBackpointers[startPos][stateIndex];
			ScoredPhrasePairForSentence trans = lexSortedTranslaitons.get(transIndex);
			//			System.out.println(trans);
			translation.add(trans);
			stateIndex = prevStateBackpointers[startPos][stateIndex];
			startPos = prevStartPos;
			score += trans.score;
		}
		//		System.out.println(max);
		Collections.reverse(translation);
		List<String> result = new ArrayList<String>();
		for (ScoredPhrasePairForSentence trans : translation) {
			result.addAll(trans.getEnglish());
		}
		final double lmScore = scoreSentence(result);
		final double hypLmScore = max - score;
		score += lmScore;
		//		if (Math.abs(lmScore - hypLmScore) > 1e-3) {
		//			System.err.println("Warning: Lm returned inconsisten
		//		}
		//		assert Math.abs(max - score) < 1e-3;
		return translation;

	}

	private double scoreSentence(List<String> sentence) {
		List<String> sentenceWithBounds = new BoundedList<String>(sentence, NgramLanguageModel.START, NgramLanguageModel.STOP);

		int lmOrder = lm.getOrder();
		double sentenceScore = 0.0;
		for (int i = 1; i < lmOrder - 1 && i <= sentenceWithBounds.size() + 1; ++i) {
			final List<String> ngram = sentenceWithBounds.subList(-1, i);
			int[] ngramArray = toArray(ngram);
			final double scoreNgram = lm.getNgramLogProbability(ngramArray, 0, ngramArray.length);
			sentenceScore += scoreNgram;
		}
		for (int i = lmOrder - 1; i < sentenceWithBounds.size() + 2; ++i) {
			final List<String> ngram = sentenceWithBounds.subList(i - lmOrder, i);
			int[] ngramArray = toArray(ngram);
			final double scoreNgram = lm.getNgramLogProbability(ngramArray, 0, ngramArray.length);
			sentenceScore += scoreNgram;
		}
		return sentenceScore;
	}

	/**
	 * @param ngram
	 * @return
	 */
	private int[] toArray(final List<String> ngram) {
		int[] ngramArray = new int[ngram.size()];
		for (int w = 0; w < ngramArray.length; ++w) {
			ngramArray[w] = lexIndexer.addAndGetIndex(ngram.get(w));
		}
		return ngramArray;
	}

	/**
	 * @param lmOrder
	 * @param length
	 * @param k
	 * @param hash
	 * @param scoreChart
	 * @param lmContexts
	 * @param lmContextLengths
	 * @param transBackpointers
	 * @param prevStateBackpointers
	 * @param startBackpointers
	 * @param beams
	 * @param start
	 * @param lexSortedTranslations
	 * @param prevState
	 * @param prevScore
	 * @param prevLmStateLength
	 * @param lmStateBuf
	 * @param transIndex
	 */
	private static void innerLoop(final int lmOrder, int length, final int k, final boolean hash, final double[][] scoreChart, final int[][][] lmContexts,
		final int[][] lmContextLengths, final int[][] transBackpointers, final int[][] prevStateBackpointers, final int[][] startBackpointers,
		final IntPriorityQueue[] beams, int start, List<ScoredPhrasePairForSentence> lexSortedTranslations, int prevState, double prevScore,
		final int prevLmStateLength, int[] lmStateBuf, int transIndex, StringIndexer lexIndexer, final NgramLanguageModel lm) {
		ScoredPhrasePairForSentence trans = lexSortedTranslations.get(transIndex);
		int[] indexedTrg = trans.english.indexedEnglish;
		System.arraycopy(indexedTrg, 0, lmStateBuf, prevLmStateLength, indexedTrg.length);
		final int newConsumedLength = start + trans.getForeignLength();
		final boolean isEnd = newConsumedLength == length;
		final int prevStatePlusPhraselength = prevLmStateLength + indexedTrg.length;
		if (isEnd) lmStateBuf[prevStatePlusPhraselength] = lexIndexer.indexOf(NgramLanguageModel.STOP);
		final int totalTrgLength = prevStatePlusPhraselength + (isEnd ? 1 : 0);
		double score = prevScore + trans.score;

		score += scoreLm(lmOrder, prevLmStateLength, lmStateBuf, totalTrgLength, lm);
		final int lmStartPos = Math.max(0, totalTrgLength - lmOrder + 1);
		final int hashIndex = hash(lmStateBuf, lmStartPos, totalTrgLength, k);
		if (hash) {
			int newStateIndex = hashIndex;
			final double[] scoreChartHere = scoreChart[newConsumedLength];
			if (score > scoreChartHere[newStateIndex]) {
				scoreChartHere[newStateIndex] = score;
				lmContextLengths[newConsumedLength][newStateIndex] = totalTrgLength - lmStartPos;
				System.arraycopy(lmStateBuf, lmStartPos, lmContexts[newConsumedLength][newStateIndex], 0, totalTrgLength - lmStartPos);

				//			lmContexts[newConsumedLength][newStateIndex] = Arrays.copyOfRange(lmStateBuf, lmStartPos, totalTrgLength);

				transBackpointers[newConsumedLength][newStateIndex] = transIndex;
				startBackpointers[newConsumedLength][newStateIndex] = start;
				prevStateBackpointers[newConsumedLength][newStateIndex] = prevState;

			}
		} else {
			doBeamUpdate(k, scoreChart, lmContexts, lmContextLengths, transBackpointers, prevStateBackpointers, startBackpointers, beams, start, prevState,
				lmStateBuf, transIndex, newConsumedLength, totalTrgLength, score, lmStartPos, hashIndex);
		}

	}

	/**
	 * @param lmOrder
	 * @param prevLmState
	 * @param lmStateBuf
	 * @param totalTrgLength
	 * @param score
	 * @return
	 */
	private static double scoreLm(final int lmOrder, final int prevLmStateLength, final int[] lmStateBuf, final int totalTrgLength, final NgramLanguageModel lm) {
		double score = 0.0;

		if (prevLmStateLength < lmOrder - 1) {
			for (int i = 1; prevLmStateLength + i < lmOrder; ++i) {
				final double lmProb = lm.getNgramLogProbability(lmStateBuf, 0, prevLmStateLength + i);
				score += lmProb;
			}
		}
		for (int i = 0; i <= totalTrgLength - lmOrder; ++i) {
			final double lmProb = lm.getNgramLogProbability(lmStateBuf, i, i + lmOrder);
			score += lmProb;
		}
		return score;
	}

	/**
	 * @param k
	 * @param scoreChart
	 * @param lmContexts
	 * @param transBackpointers
	 * @param prevStateBackpointers
	 * @param startBackpointers
	 * @param startBackpointers2
	 * @param beams
	 * @param start
	 * @param prevState
	 * @param lmStateBuf
	 * @param transIndex
	 * @param newConsumedLength
	 * @param totalTrgLength
	 * @param score
	 * @param lmStartPos
	 * @param hashIndex
	 */
	private static void doBeamUpdate(final int k, final double[][] scoreChart, final int[][][] lmContexts, final int[][] lmContextLengths,
		final int[][] transBackpointers, final int[][] prevStateBackpointers, final int[][] startBackpointers, final IntPriorityQueue[] beams, int start,
		int prevState, int[] lmStateBuf, int transIndex, final int newConsumedLength, final int totalTrgLength, double score, final int lmStartPos,
		final int hashIndex) {
//		assert hasMinusOne(lmContextLengths[newConsumedLength]) > 3;
//		if (newConsumedLength == 5) {
//			System.out.println(hasMinusOne(lmContextLengths[newConsumedLength]));
//			
//		}
		int newStateIndex = findStateIndex(lmStateBuf, lmStartPos, totalTrgLength, lmContexts[newConsumedLength], lmContextLengths[newConsumedLength],
			hashIndex);
		
		IntPriorityQueue beam = beams[newConsumedLength];
		final double cost = -1.0 * score;
		if (beam.size() < k || cost < beam.getPriorityOfBest()) {
			double oldCost = beam.getPriorityOfElement(newStateIndex);
			final boolean isNaN = Double.isNaN(oldCost);
			final boolean less = cost < oldCost;
			if (isNaN || less) {
				if (isNaN) {
					beam.put(newStateIndex, cost);
					
					if (beam.size() > k) {
						int indexWhichFellOfBeam = beam.next();
						lmContextLengths[newConsumedLength][indexWhichFellOfBeam] = -1;
						scoreChart[newConsumedLength][indexWhichFellOfBeam] = Double.NEGATIVE_INFINITY;

						transBackpointers[newConsumedLength][indexWhichFellOfBeam] = -1;
						startBackpointers[newConsumedLength][indexWhichFellOfBeam] = -1;
						prevStateBackpointers[newConsumedLength][indexWhichFellOfBeam] = -1;
					}
				} else {
					assert lmContextLengths[newConsumedLength][newStateIndex] != -1;
					beam.increaseKey(newStateIndex, cost);
				}
				transBackpointers[newConsumedLength][newStateIndex] = transIndex;
				startBackpointers[newConsumedLength][newStateIndex] = start;
				prevStateBackpointers[newConsumedLength][newStateIndex] = prevState;

				lmContextLengths[newConsumedLength][newStateIndex] = totalTrgLength - lmStartPos;
//				assert hasMinusOne(lmContextLengths[newConsumedLength]) > 3;
				System.arraycopy(lmStateBuf, lmStartPos, lmContexts[newConsumedLength][newStateIndex], 0, totalTrgLength - lmStartPos);
				lmContexts[newConsumedLength][newStateIndex] = CollectionUtils.copyOfRange(lmStateBuf, lmStartPos, totalTrgLength);
				scoreChart[newConsumedLength][newStateIndex] = score;
//				assert beamAndContextMatch(beam, lmContextLengths[newConsumedLength]);

			}
		}
	}

	private static boolean beamAndContextMatch(IntPriorityQueue beam, int[] is) {
		if (true) return true;
		for (Entry<Integer, Double> entry : beam.asCounter().getEntrySet()) {
			if (is[entry.getKey()] == -1) {
				@SuppressWarnings("unused")
				int x = 5;
			}
		}
		for (int i = 0; i < is.length; ++i) {
			if (is[i] == -1) {
				assert Double.isNaN(beam.getPriorityOfElement(i));
			}
		}
		return true;
	}

	private static int hasMinusOne(int[] is) {
		
		int x = 0;
		for (int i : is)
			if (i < 0) x++;
		return x;
	}

	private static int findStateIndex(int[] lmStateBuf, int start, int end, int[][] lmContexts, int[] lmContextLengths, final int hashIndex) {
		int currIndex = hashIndex;
		while (true) {
			final int lmContextLength = lmContextLengths[currIndex];
			if (lmContextLength < 0 || equals(lmContexts[currIndex], lmContextLength, lmStateBuf, start, end)) { return currIndex; }
			currIndex++;
			if (currIndex == lmContexts.length) currIndex = 0;
		}
	}

	private static boolean equals(final int[] a, final int length, final int[] b, final int start, final int end) {
		if (end - start != length) return false;
		for (int i = 0; i < length; ++i) {
			if (a[i] != b[start + i]) return false;
		}
		return true;
	}

	private static int hash(int[] lmStateBuf, int start, int end, int k) {
		int hash = MurmurHash.hash32(lmStateBuf, start, end, 13);
		if (hash < 0) hash = -hash;
		return (hash % k);
	}

	private static void fill(double[][] a, double val) {
		for (int i = 0; i < a.length; i++) {
			Arrays.fill(a[i], val);
		}
	}

	private static void fill(int[][] a, int val) {
		for (int i = 0; i < a.length; i++) {
			Arrays.fill(a[i], val);
		}
	}

	private static void fill(double[][] a, int until1, int until2, double val) {
		for (int i = 0; i < until1; ++i) {
			Arrays.fill(a[i], 0, until2 == Integer.MAX_VALUE ? a[i].length : until2, val);
		}
	}

}
