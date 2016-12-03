package edu.berkeley.nlp.mt.phrasetable;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import edu.berkeley.nlp.assignments.align.AlignmentTester;
import edu.berkeley.nlp.assignments.align.AlignmentTester.IntArrayWrapper;
import edu.berkeley.nlp.assignments.align.AlignmentTester.IntPhrasePair;
import edu.berkeley.nlp.io.IOUtils;
import edu.berkeley.nlp.mt.decoder.Logger;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.StrUtils;
import edu.berkeley.nlp.util.StringIndexer;
import edu.berkeley.nlp.util.TIntOpenHashMap;

/**
 * Stores phrase pairs and their scores. Inside a decoder, you should call
 * initialize() to get back an object which returns translations for a specific
 * sentence.
 * 
 * @author adampauls
 * 
 */
public class PhraseTable
{

	private int maxPhraseSize;

	private int maxNumTranslations;

	private static class PhrasePair
	{
		/**
		 * @param foreign
		 * @param english
		 */
		public PhrasePair(String[] foreign, EnglishPhrase english) {
			super();
			this.foreign = foreign;
			this.english = english;
		}

		String[] foreign;

		EnglishPhrase english;
	}

	Map<List<String>, List<ScoredPhrase>> table;

	public static final String[] MOSES_FEATURE_NAMES = new String[] { "P(f|e)", "lex(f|e)", "P(e|f)", "lex(e|f)", "bias", "wordBonus" };

	private static final int P_F_GIVEN_E = 0;

	private static final int P_E_GIVEN_F = 2;

	private static final int BIAS = 4;

	private static final int WORD_BONUS = 5;

	/**
	 * 
	 * @param maxPhraseSize
	 *            The maximum length of either side of a phrase
	 * @param maxNumTranslations
	 *            The maximum number of translations per foreign span.
	 */
	public PhraseTable(int maxPhraseSize, int maxNumTranslations) {
		this.maxPhraseSize = maxPhraseSize;
		this.maxNumTranslations = maxNumTranslations;
	}

	public PhraseTableForSentence initialize(List<String> sentence) {
		return new PhraseTableForSentence(this, sentence);
	}

	public int getMaxPhraseSize() {
		return maxPhraseSize;
	}

	public int getMaxNumTranslations() {
		return maxNumTranslations;
	}

	public void readFromFile(String file, Counter<String> featureWeights) {

		initStorage();
		Logger.startTrack("Reading phrase table from file " + file);
		int l = 0;

		try {
			for (String line : CollectionUtils.iterable(IOUtils.lineIterator(file))) {
				l++;
				if (l % 100000 == 0) System.out.println("Line " + l);
				float[] features = new float[6];
				PhrasePair phrasePair = readMosesRule(line, features);

				if (phrasePair.english.indexedEnglish.length > maxPhraseSize) continue;
				if (phrasePair.foreign.length > maxPhraseSize) continue;
				ScoredPhrase t = new ScoredPhrase(phrasePair.english, getFeatureCounter(features).dotProduct(featureWeights));

				addTranslation(t, Arrays.asList(phrasePair.foreign));

			}
		} catch (IOException e) {
			throw new RuntimeException(e);

		}

		sortTranslations();
		Logger.endTrack();
	}

	public void readFromCounts(HashMap<IntPhrasePair, Integer> counter, HashMap<IntArrayWrapper, Integer> eCounter, HashMap<IntArrayWrapper, Integer> fCounter,
		Counter<String> featureWeights, StringIndexer fWordIndexer) {

		initStorage();
		Logger.startTrack("Reading phrase table from counts");
		int l = 0;

		for (Entry<IntPhrasePair, Integer> entry : counter.entrySet()) {
			l++;
			if (l % 100000 == 0) System.out.println("Phrase " + l);
			int[] english = entry.getKey().english;
			int[] foreign = entry.getKey().foreign;
			float[] features = new float[6];
			features[P_F_GIVEN_E] = -(float) Math.log(entry.getValue() / (double) eCounter.get(new AlignmentTester.IntArrayWrapper(english)));
			features[P_E_GIVEN_F] = -(float) Math.log(entry.getValue() / (double) fCounter.get(new AlignmentTester.IntArrayWrapper(foreign)));

			if (english.length > maxPhraseSize) continue;
			if (foreign.length > maxPhraseSize) continue;
			ScoredPhrase t = new ScoredPhrase(new EnglishPhrase(english), getFeatureCounter(features).dotProduct(featureWeights));

			addTranslation(t, Arrays.asList(toStringArray(foreign, fWordIndexer)));

		}

		sortTranslations();
		Logger.endTrack();
	}

	private String[] toStringArray(int[] foreign, StringIndexer fWordIndexer) {
		String[] stringArray = new String[foreign.length];
		for (int i = 0; i < foreign.length; ++i) {
			stringArray[i] = fWordIndexer.get(foreign[i]);
		}
		return stringArray;
	}

	List<ScoredPhrase> getTranslationsFor(List<String> subList) {
		return table.get(subList);
	}

	private Counter<String> getFeatureCounter(float[] features) {
		Counter<String> ret = new Counter<String>();
		for (int i = 0; i < features.length; ++i) {
			ret.setCount(MOSES_FEATURE_NAMES[i], features[i]);
		}
		return ret;
	}

	private PhrasePair readMosesRule(String ruleString, float[] features) {
		String[] parts = ruleString.trim().split("\\|\\|\\|");
		assert (parts.length == 3 || parts.length == 5);
		if (parts.length == 5) parts[2] = parts[4];
		final String[] srcArray = parts[0].trim().split(" ");
		final String[] trgArray = parts[1].trim().split(" ");
		intern(srcArray);
		intern(trgArray);

		String[] featStrings = parts[2].trim().split("\\s+");
		for (int i = 0; i < featStrings.length; i++) {

			try {
				Double val = Double.parseDouble(featStrings[i]);
				if (val.isInfinite() || val.isNaN()) {
					Logger.warn("Non-finite feature: " + featStrings[i]);
					continue;
				}
				val = -Math.log(val);

				features[i] = val.floatValue();
			} catch (NumberFormatException n) {
				Logger.warn("Feature syntax error: " + featStrings[i]);
			}
		}
		features[5] = trgArray.length;
		return new PhrasePair(srcArray, new EnglishPhrase(trgArray));

	}

	private void intern(String[] a) {
		for (int i = 0; i < a.length; ++i)
			a[i] = a[i].intern();
	}

	private void initStorage() {
		table = new HashMap<List<String>, List<ScoredPhrase>>();
	}

	private void addTranslation(ScoredPhrase t, List<String> foreign) {
		CollectionUtils.addToValueList(table, foreign, t);
	}

	private void sortTranslations() {

		for (Entry<List<String>, List<ScoredPhrase>> entry : table.entrySet()) {

			Collections.sort(entry.getValue(), new Comparator<ScoredPhrase>()
			{

				public int compare(ScoredPhrase o1, ScoredPhrase o2) {
					return Double.compare(o2.score, o1.score);
				}
			});
		}
	}

	public void writeToFile(String phraseTableOut) {
		Logger.startTrack("Writing phrase table to " + phraseTableOut);
		PrintWriter out = IOUtils.openOutHard(phraseTableOut);
		int k = 0;
		for (Entry<List<String>, List<ScoredPhrase>> entry : table.entrySet()) {
			if (k % 100000 == 0) System.out.println("Phrase " + k);
			k++;
			for (ScoredPhrase p : entry.getValue()) {
				out.println(StrUtils.join(entry.getKey()) + " ||| " + StrUtils.join(p.getEnglish()) + " ||| " + p.score);
			}

		}
		out.close();
		Logger.endTrack();
	}

}