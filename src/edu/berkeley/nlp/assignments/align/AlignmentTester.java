package edu.berkeley.nlp.assignments.align;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.assignments.align.student.HeuristicAlignerFactory;
import edu.berkeley.nlp.assignments.align.student.HmmAlignerFactory;
import edu.berkeley.nlp.assignments.align.student.Model1AlignerFactory;
import edu.berkeley.nlp.io.IOUtils;
import edu.berkeley.nlp.io.SentenceCollection;
import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.langmodel.RandomLanguageModel;
import edu.berkeley.nlp.langmodel.StubLanguageModel;
import edu.berkeley.nlp.langmodel.impl.KneserNeyLm;
import edu.berkeley.nlp.langmodel.impl.NgramLanguageModelAdaptor;
import edu.berkeley.nlp.langmodel.impl.NgramMapOpts;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.BaselineWordAligner;
import edu.berkeley.nlp.mt.BleuScore;
import edu.berkeley.nlp.mt.PhraseExtractor;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.Weights;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.mt.WordAlignerFactory;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.Decoder.StaticMethods;
import edu.berkeley.nlp.mt.decoder.DecoderFactory;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.decoder.LinearDistortionModel;
import edu.berkeley.nlp.mt.decoder.Logger;
import edu.berkeley.nlp.mt.decoder.MonotonicGreedyDecoder.MonotonicGreedyDecoderFactory;
import edu.berkeley.nlp.mt.decoder.internal.BeamDecoder;
import edu.berkeley.nlp.mt.decoder.StubDistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.CommandLineUtils;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.MemoryUsageUtils;
import edu.berkeley.nlp.util.Pair;
import edu.berkeley.nlp.util.StrUtils;
import edu.berkeley.nlp.util.StringIndexer;
import edu.berkeley.nlp.util.TIntOpenHashMap;
import edu.berkeley.nlp.util.TIntOpenHashMap.Entry;

/**
 * @author Adam Pauls
 */
public class AlignmentTester
{

	enum AlignerType
	{
		HEURISTIC
		{

			@Override
			public WordAlignerFactory getWordAlignerFactory() {
				return new HeuristicAlignerFactory();
			}

		},
		MODEL1
		{

			@Override
			public WordAlignerFactory getWordAlignerFactory() {
				return new Model1AlignerFactory();
			}

		},
		HMM
		{

			@Override
			public WordAlignerFactory getWordAlignerFactory() {
				return new HmmAlignerFactory();
			}

		},
		BASELINE
		{
			@Override
			public WordAlignerFactory getWordAlignerFactory() {
				return new BaselineWordAligner.Factory();
			}
		};

		public abstract WordAlignerFactory getWordAlignerFactory();
	}

	public static void main(String[] args) {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		// You can use this to make decoding runs run in less time, but remember that we will
		// evaluate you on all test sentences.
		int maxNumTest = Integer.MAX_VALUE;
		boolean sanityCheck = false;
		boolean printTranslations = true;
		boolean randomLm = false;

		int maxTrainingSentences = Integer.MAX_VALUE;
		if (argMap.containsKey("-maxTrain")) {
			maxTrainingSentences = Integer.parseInt(argMap.get("-maxTrain"));
		}

		if (argMap.containsKey("-maxNumDecode")) {
			maxNumTest = Integer.parseInt(argMap.get("-maxNumDecode"));
		}
		System.out.println("Decoding " + (maxNumTest == Integer.MAX_VALUE ? "all" : ("" + maxNumTest)) + " sentences.");

		if (argMap.containsKey("-noprint")) {
			printTranslations = false;
		}

		boolean justAlign = false;
		if (argMap.containsKey("-justAlign")) {
			justAlign = true;
		}

		String phraseTableOut = null;
		if (argMap.containsKey("-phraseTableOut")) {
			phraseTableOut = argMap.get("-phraseTableOut");
		}

		boolean printAlignments = false;
		if (argMap.containsKey("-printAlignments")) {
			printAlignments = true;
		}

		if (argMap.containsKey("-sanityCheck")) {
			sanityCheck = true;
		}
		if (sanityCheck) System.out.println("Only doing sanity check.");

		// Use an LM which just returns a random (but consistent for a given n-gram) score. This should speed up loading and facilitate sanity checking
		if (argMap.containsKey("-randomLm")) {
			randomLm = true;
		}

		if (sanityCheck) randomLm = true;
		String prefix = sanityCheck ? "sanity_" : "";

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		String dataset = "miniTest";
		if (argMap.containsKey("-data")) {
			dataset = argMap.get("-data");
			System.out.println("Running with data: " + dataset);
		} else if (!sanityCheck) {
			System.out.println("No data set specified.  Use -data [miniTest, validate, test].");
		}

		Iterable<SentencePair> trainingSentencePairs = SentencePair.readSentencePairs(new File(basePath, prefix + "training").getPath(), maxTrainingSentences);
		List<SentencePair> testSentencePairs = new ArrayList<SentencePair>();
		Map<Integer, Alignment> testAlignments = new HashMap<Integer, Alignment>();
		if (sanityCheck) {
			testSentencePairs = toList(SentencePair.readSentencePairs(basePath + "/test_aligns_sanity", Integer.MAX_VALUE));
			testAlignments = Alignment.readAlignments(basePath + "/test_aligns_sanity/sanity_test.wa");
		} else if (dataset.equalsIgnoreCase("test")) {
			testSentencePairs = toList(SentencePair.readSentencePairs(basePath + "/test_aligns_big", Integer.MAX_VALUE));
			testAlignments = Alignment.readAlignments(basePath + "/test_aligns_big/test.wa");
		} else if (dataset.equalsIgnoreCase("validate")) {
			testSentencePairs = toList(SentencePair.readSentencePairs(basePath + "/test_aligns_validate", Integer.MAX_VALUE));
			testAlignments = Alignment.readAlignments(basePath + "/test_aligns_validate/trial.wa");
		} else if (dataset.equalsIgnoreCase("miniTest")) {
			testSentencePairs = toList(SentencePair.readSentencePairs(basePath + "/test_aligns_mini", Integer.MAX_VALUE));
			testAlignments = Alignment.readAlignments(basePath + "/test_aligns_mini/mini.wa");
		} else {
			throw new RuntimeException("Bad data set mode: " + dataset + ", use test, validate, or miniTest.");
		}
		Iterable<SentencePair> concatSentencePairs = CollectionUtils.concat(trainingSentencePairs, testSentencePairs);

		AlignerType wordAlignerType = AlignerType.BASELINE;
		if (argMap.containsKey("-alignerType")) {
			wordAlignerType = AlignerType.valueOf(argMap.get("-alignerType"));
		}
		System.out.println("Using alignerType: " + wordAlignerType);

		// Read in all the assignment data
		File lmFile = new File(basePath, prefix + "lm.gz");

		File testFrench = new File(basePath, prefix + "test.fr");
		File testEnglish = new File(basePath, prefix + "test.en");
		File weightsFile = new File(basePath, "weights.txt");

		final Counter<String> weights = Weights.readWeightsFile(weightsFile);
		PhraseTable phraseTable = alignAndBuildPhraseTable(justAlign, printAlignments, trainingSentencePairs, testSentencePairs, testAlignments,
			concatSentencePairs, wordAlignerType, weights);
		if (phraseTableOut != null) {
			phraseTable.writeToFile(phraseTableOut);
		}
		NgramLanguageModel languageModel = getActualLanguageModel(lmFile, randomLm);

		Decoder decoder = new BeamDecoder(languageModel, phraseTable, EnglishWordIndexer.getIndexer());
		//		final DistortionModel distortionModel = decoderType.getDistortionModel(weights);
		evaluateDecoder(decoder, phraseTable, testFrench, testEnglish, weightsFile, getActualLanguageModel(lmFile, randomLm), maxNumTest, printTranslations,
			getActualDistortionModel(weights));

	}

	/**
	 * @param justAlign
	 * @param printAlignments
	 * @param trainingSentencePairs
	 * @param testSentencePairs
	 * @param testAlignments
	 * @param concatSentencePairs
	 * @param wordAlignerType
	 * @param weights
	 * @return
	 */
	private static PhraseTable alignAndBuildPhraseTable(boolean justAlign, boolean printAlignments, Iterable<SentencePair> trainingSentencePairs,
		List<SentencePair> testSentencePairs, Map<Integer, Alignment> testAlignments, Iterable<SentencePair> concatSentencePairs, AlignerType wordAlignerType,
		final Counter<String> weights) {
		// Build model
		Logger.startTrack("Building aligner");
		WordAligner wordAligner = wordAlignerType.getWordAlignerFactory().newAligner(concatSentencePairs);
		Logger.endTrack();
		testAlignments(wordAligner, testSentencePairs, testAlignments, printAlignments);

		if (justAlign) System.exit(0);
		PhraseTable phraseTable = getPhraseTableFromAlignedCorpus(trainingSentencePairs, wordAligner, weights);
		return phraseTable;
	}

	public static class IntArrayWrapper
	{
		/**
		 * @param array
		 */
		public IntArrayWrapper(int[] array) {
			super();
			this.array = array;
		}

		@Override
		public int hashCode() {
			if (array == null) return 0;

			int result = 1;
			for (int element : array)
				result = 79 * result + element;

			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj) return true;
			if (obj == null) return false;
			if (getClass() != obj.getClass()) return false;
			IntArrayWrapper other = (IntArrayWrapper) obj;
			if (!Arrays.equals(array, other.array)) return false;
			return true;
		}

		private int[] array;
	}

	public static class IntPhrasePair
	{
		/**
		 * @param foreign
		 * @param english
		 */
		public IntPhrasePair(int[] foreign, int[] english) {
			super();
			this.foreign = foreign;
			this.english = english;
		}

		@Override
		public int hashCode() {
			final int prime = 57;
			int result = 1;
			result = prime * result + Arrays.hashCode(english);
			result = prime * result + Arrays.hashCode(foreign);
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj) return true;
			if (obj == null) return false;
			if (getClass() != obj.getClass()) return false;
			IntPhrasePair other = (IntPhrasePair) obj;
			if (!Arrays.equals(english, other.english)) return false;
			if (!Arrays.equals(foreign, other.foreign)) return false;
			return true;
		}

		public int[] foreign;

		public int[] english;

	}

	public static Iterable<SentencePair> reverse(final Iterable<SentencePair> data) {
		return new Iterable<SentencePair>()
		{

			public Iterator<SentencePair> iterator() {
				return new CollectionUtils.Transform<SentencePair, SentencePair>(data.iterator())
				{

					@Override
					protected SentencePair transform(SentencePair sp) {
						return sp.getReversedCopy();
					}

				};

			}
		};
	}

	private static PhraseTable getPhraseTableFromAlignedCorpus(Iterable<SentencePair> trainingSentencePairs, WordAligner wordAligner, Counter<String> weights) {
		Logger.startTrack("Extracting phrases");
		final int maxPhraseSize = 5;
		final int maxNumTranslations = 30;
		final int maxNumUnaligned = 1;
		StringIndexer foreignWordIndexer = new StringIndexer();
		StringIndexer englishWordIndexer = EnglishWordIndexer.getIndexer();
		HashMap<IntPhrasePair, Integer> counter = new HashMap<IntPhrasePair, Integer>(1000, 0.5f);
		int sent = 0;
		for (SentencePair sentencePair : trainingSentencePairs) {
			if (sent % 1000 == 0) System.out.println("Sentence " + sent);
			sent++;
			final PhraseExtractor phraseExtractor = new PhraseExtractor();
			Alignment al = wordAligner.alignSentencePair(sentencePair);
			List<IntPhrasePair> phrases = phraseExtractor.extract(al, lowercase(sentencePair), maxPhraseSize, englishWordIndexer, foreignWordIndexer,
				maxNumUnaligned);
			for (IntPhrasePair pair : phrases) {
				increment(counter, pair, 1);
			}

		}
		Logger.endTrack();
		HashMap<IntArrayWrapper, Integer> eCounter = new HashMap<IntArrayWrapper, Integer>(1000, 0.5f);
		HashMap<IntArrayWrapper, Integer> fCounter = new HashMap<IntArrayWrapper, Integer>(1000, 0.5f);

		Logger.startTrack("Counting extracted phrases");
		int phraseNum = 0;
		for (java.util.Map.Entry<IntPhrasePair, Integer> pair : counter.entrySet()) {
			if (phraseNum % 100000 == 0) System.out.println("Phrase " + phraseNum + " of " + counter.size());
			phraseNum++;
			int count = pair.getValue();
			IntPhrasePair key = pair.getKey();

			increment(eCounter, new IntArrayWrapper(key.english), count);
			increment(fCounter, new IntArrayWrapper(key.foreign), count);
		}
		Logger.endTrack();
		PhraseTable phraseTable = new PhraseTable(maxPhraseSize, maxNumTranslations);
		phraseTable.readFromCounts(counter, eCounter, fCounter, weights, foreignWordIndexer);
		return phraseTable;
	}

	/**
	 * @param counter
	 * @param pair
	 */
	private static <T> void increment(HashMap<T, Integer> counter, T pair, int count) {
		final Integer integer = counter.get(pair);
		int old = integer == null ? 0 : integer;
		counter.put(pair, old + count);
	}

	private static SentencePair lowercase(SentencePair sentencePair) {
		SentencePair ret = new SentencePair(sentencePair.getSentenceID(), sentencePair.getSourceFile(), lowercase(sentencePair.getEnglishWords()),
			lowercase(sentencePair.getFrenchWords()));
		return ret;
	}

	private static List<SentencePair> toList(Iterable<SentencePair> sentencePairs) {

		List<SentencePair> list = new ArrayList<SentencePair>();
		for (SentencePair sp : sentencePairs) {
			list.add(sp);
		}
		return list;
	}

	/**
	 * @param useTestSet
	 * @param phraseTableFile
	 * @param devFrench
	 * @param devEnglish
	 * @param testFrench
	 * @param testEnglish
	 * @param weightsFile
	 * @param languageModel
	 */
	private static void evaluateDecoder(Decoder decoder, PhraseTable phraseTable, File testFrench, File testEnglish, File weightsFile,
		NgramLanguageModel languageModel, int maxNumTest, boolean printTranslations, DistortionModel distortionModel) {

		MemoryUsageUtils.printMemoryUsage();
		final String frenchData = (testFrench).getPath();
		Iterable<List<String>> frenchSentences = lowercase(SentenceCollection.Reader.readSentenceCollection(frenchData));
		final String englishData = (testEnglish).getPath();
		Iterable<List<String>> englishSentences = lowercase(SentenceCollection.Reader.readSentenceCollection(englishData));
		List<BleuScore> scores = new ArrayList<BleuScore>();
		double[] modelScore = new double[1];
		doDecoding(decoder, frenchSentences, englishSentences, scores, maxNumTest, printTranslations, languageModel, modelScore, distortionModel);
		String bleuString = new BleuScore(scores).toString();
		System.out.println("BLEU score on " + ("test") + " data was " + bleuString);
		System.out.println("Total model score on " + ("test") + " data was " + modelScore[0]);

	}

	private static Iterable<List<String>> lowercase(final Iterable<List<String>> readSentenceCollection) {
		return new Iterable<List<String>>()
		{
			public Iterator<List<String>> iterator() {
				return new CollectionUtils.Transform<List<String>, List<String>>(readSentenceCollection.iterator())
				{
					@Override
					protected List<String> transform(List<String> next) {
						return lowercase(next);
					}

				};
			}
		};
	}

	/**
	 * @param next
	 * @return
	 */
	private static List<String> lowercase(List<String> next) {
		List<String> ret = new ArrayList<String>(next.size());
		for (String s : next)
			ret.add(s.toLowerCase());
		return ret;
	}

	/**
	 * @param weights
	 * @return
	 */
	private static LinearDistortionModel getActualDistortionModel(Counter<String> weights) {
		return new LinearDistortionModel(4, weights.getCount("linearDist"));
	}

	private static NgramLanguageModel actualLanguageModel = null;

	/**
	 * @param lmFile
	 * @param random
	 * @return
	 */
	private static NgramLanguageModel getActualLanguageModel(File lmFile, boolean random) {
		if (actualLanguageModel == null)
			actualLanguageModel = random ? new RandomLanguageModel() : new NgramLanguageModelAdaptor(KneserNeyLm.fromFile(new NgramMapOpts(), lmFile.getPath(),
				3, EnglishWordIndexer.getIndexer()));
		return actualLanguageModel;
	}

	/**
	 * @param decoder
	 * @param frenchSentences
	 * @param englishSentences
	 * @param scores
	 * @param languageModel
	 */
	private static void doDecoding(Decoder decoder, Iterable<List<String>> frenchSentences, Iterable<List<String>> englishSentences, List<BleuScore> scores,
		int maxNumTest, boolean printTranslations, NgramLanguageModel languageModel, double[] modelScore, DistortionModel dm) {
		long startTime = System.nanoTime();
		int sent = 0;
		System.out.println("Decoding " + (maxNumTest == Integer.MAX_VALUE ? "all" : ("" + maxNumTest)) + " test sentences");
		for (Pair<List<String>, List<String>> input : CollectionUtils.zip(Pair.makePair(frenchSentences, englishSentences))) {
			if (sent >= maxNumTest) break;
			sent++;

			if (sent % 100 == 0) Logger.logs("On sentence " + sent);

			final List<ScoredPhrasePairForSentence> hyp = decoder.decode(input.getFirst());
			double score = StaticMethods.scoreHypothesis(hyp, languageModel, dm);
			List<String> hypothesisEnglish = Decoder.StaticMethods.extractEnglish(hyp);
			List<String> reference = input.getSecond();
			if (printTranslations) {

				System.out.println("Model score:\t" + score);
				System.out.println("Input:\t\t" + StrUtils.join(input.getFirst()));
				System.out.println("Hypothesis\t" + StrUtils.join(hypothesisEnglish));
				System.out.println("Reference:\t" + StrUtils.join(reference));
				System.out.println();
			}
			BleuScore bleuScore = new BleuScore(hypothesisEnglish, reference);
			modelScore[0] += score;
			scores.add(bleuScore);
		}
		long endTime = System.nanoTime();

		System.out.println("Decoding took " + BleuScore.formatDouble((endTime - startTime) / 1e9) + "s");
	}

	private static void testAlignments(WordAligner wordAligner, List<SentencePair> testSentencePairs, Map<Integer, Alignment> testAlignments, boolean verbose) {
		int proposedSureCount = 0;
		int proposedPossibleCount = 0;
		int sureCount = 0;
		int proposedCount = 0;
		for (SentencePair sentencePair : testSentencePairs) {
			Alignment proposedAlignment = wordAligner.alignSentencePair(sentencePair);
			Alignment referenceAlignment = testAlignments.get(sentencePair.getSentenceID());
			if (referenceAlignment == null) throw new RuntimeException("No reference alignment found for sentenceID " + sentencePair.getSentenceID());
			if (verbose) System.out.println("Alignment:\n" + Alignment.render(referenceAlignment, proposedAlignment, sentencePair));
			for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
				for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
					boolean proposed = proposedAlignment.containsSureAlignment(englishPosition, frenchPosition);
					boolean sure = referenceAlignment.containsSureAlignment(englishPosition, frenchPosition);
					boolean possible = referenceAlignment.containsPossibleAlignment(englishPosition, frenchPosition);
					if (proposed && sure) proposedSureCount += 1;
					if (proposed && possible) proposedPossibleCount += 1;
					if (proposed) proposedCount += 1;
					if (sure) sureCount += 1;
				}
			}
		}
		System.out.println("Precision: " + proposedPossibleCount / (double) proposedCount);
		System.out.println("Recall: " + proposedSureCount / (double) sureCount);
		System.out.println("AER: " + (1.0 - (proposedSureCount + proposedPossibleCount) / (double) (sureCount + proposedCount)));
	}

}
