package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.assignments.align.student.aligner.*;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.mt.WordAlignerFactory;

public class HmmAlignerFactory implements WordAlignerFactory
{

	public WordAligner newAligner(Iterable<SentencePair> trainingData) {
		return new HMMAligner(trainingData);
//		return new HMMForwardAligner(trainingData);
//		return new HMMBackwardAligner(trainingData);
	}
}
