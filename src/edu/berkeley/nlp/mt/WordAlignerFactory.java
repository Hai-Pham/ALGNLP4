package edu.berkeley.nlp.mt;

public interface WordAlignerFactory
{
	WordAligner newAligner(Iterable<SentencePair> trainingData);

}
