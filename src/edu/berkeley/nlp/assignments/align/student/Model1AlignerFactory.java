package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.assignments.align.student.aligner.IBM1Aligner;
import edu.berkeley.nlp.assignments.align.student.aligner.IBM1Aligner2;
import edu.berkeley.nlp.assignments.align.student.aligner.IBM1BackwardAligner;
import edu.berkeley.nlp.assignments.align.student.aligner.IBM1ForwardAligner;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.mt.WordAlignerFactory;

public class Model1AlignerFactory implements WordAlignerFactory
{
	public WordAligner newAligner(Iterable<SentencePair> trainingData) {
//		return new IBM1Aligner(trainingData);
		return new IBM1Aligner2(trainingData);
//		return new IBM1ForwardAligner(trainingData);
//		return new IBM1BackwardAligner(trainingData);
	}
}
