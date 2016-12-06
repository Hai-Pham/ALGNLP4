package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.assignments.align.student.aligner.HeuristicAligner;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.mt.WordAlignerFactory;

public class HeuristicAlignerFactory implements WordAlignerFactory
{

	public WordAligner newAligner(Iterable<SentencePair> trainingData) {

		 return new HeuristicAligner(trainingData);
	}



}
