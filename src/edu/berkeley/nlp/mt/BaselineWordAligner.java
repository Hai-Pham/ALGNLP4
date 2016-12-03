package edu.berkeley.nlp.mt;

/**
 * Simple alignment baseline which maps french positions to english positions.
 * If the french sentence is longer, all final word map to null.
 */
public class BaselineWordAligner implements WordAligner
{
	public static class Factory implements WordAlignerFactory
	{

		public WordAligner newAligner(Iterable<SentencePair> trainingData) {
			return new BaselineWordAligner();
		}

	}

	public Alignment alignSentencePair(SentencePair sentencePair) {
		Alignment alignment = new Alignment();
		int numFrenchWords = sentencePair.getFrenchWords().size();
		int numEnglishWords = sentencePair.getEnglishWords().size();
		for (int frenchPosition = 0; frenchPosition < numFrenchWords; frenchPosition++) {
			int englishPosition = frenchPosition;
			if (englishPosition < numEnglishWords) alignment.addAlignment(englishPosition, frenchPosition, true);
		}
		return alignment;
	}
}