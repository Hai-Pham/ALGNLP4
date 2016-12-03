package edu.berkeley.nlp.mt;

public interface WordAligner
{
	Alignment alignSentencePair(SentencePair sentencePair);
}