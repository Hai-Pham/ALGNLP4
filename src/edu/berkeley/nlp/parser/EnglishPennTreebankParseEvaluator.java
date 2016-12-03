package edu.berkeley.nlp.parser;

import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;

import java.util.*;
import java.io.PrintWriter;
import java.io.StringReader;

/**
 * Evaluates precision and recall for English Penn Treebank parse trees. NOTE:
 * Unlike the standard evaluation, multiplicity over each span is ignored. Also,
 * punction is NOT currently deleted properly (approximate hack), and other
 * normalizations (like AVDP ~ PRT) are NOT done.
 * 
 * @author Dan Klein
 */
public class EnglishPennTreebankParseEvaluator<L>
{
	abstract static class AbstractEval<L, O>
	{

		protected String str = "";

		private int exact = 0;

		private int total = 0;

		private int correctEvents = 0;

		private int guessedEvents = 0;

		private int goldEvents = 0;

		abstract Set<O> makeObjects(Tree<L> tree);

		public void evaluate(Tree<L> guess, Tree<L> gold) {
			evaluate(guess, gold, new PrintWriter(System.out, true));
		}

		/*
		 * evaluates precision and recall by calling makeObjects() to make a set
		 * of structures for guess Tree and gold Tree, and compares them with
		 * each other.
		 */
		public void evaluate(Tree<L> guess, Tree<L> gold, PrintWriter pw) {
			evaluateHelp(guess, gold, pw);

		}

		public double evaluateF1(Tree<L> guess, Tree<L> gold) {
			return evaluateHelp(guess, gold, null);
		}

		/**
		 * Returns f1
		 * 
		 * @param guess
		 * @param gold
		 * @param pw
		 */
		private double evaluateHelp(Tree<L> guess, Tree<L> gold, PrintWriter pw) {
			Set<O> guessedSet = makeObjects(guess);
			Set<O> goldSet = makeObjects(gold);
			Set<O> correctSet = new HashSet<O>();
			correctSet.addAll(goldSet);
			correctSet.retainAll(guessedSet);

			correctEvents += correctSet.size();
			guessedEvents += guessedSet.size();
			goldEvents += goldSet.size();

			int currentExact = 0;
			if (correctSet.size() == guessedSet.size() && correctSet.size() == goldSet.size()) {
				exact++;
				currentExact = 1;
			}
			total++;

			//      guess.pennPrint(pw);
			//      gold.pennPrint(pw);
			double f1 = displayPRF(str + " [Current] ", correctSet.size(), guessedSet.size(), goldSet.size(), currentExact, 1, pw);
			return f1;
		}

		/**
		 * returns F1
		 * 
		 * @param prefixStr
		 * @param correct
		 * @param guessed
		 * @param gold
		 * @param exact
		 * @param total
		 * @param pw
		 * @return
		 */
		private double displayPRF(String prefixStr, int correct, int guessed, int gold, int exact, int total, PrintWriter pw) {
			double precision = (guessed > 0 ? correct / (double) guessed : 1.0);
			double recall = (gold > 0 ? correct / (double) gold : 1.0);
			double f1 = (precision > 0.0 && recall > 0.0 ? 2.0 / (1.0 / precision + 1.0 / recall) : 0.0);

			double exactMatch = exact / (double) total;

			String displayStr = " P: " + ((int) (precision * 10000)) / 100.0 + " R: " + ((int) (recall * 10000)) / 100.0 + " F1: " + ((int) (f1 * 10000))
				/ 100.0 + " EX: " + ((int) (exactMatch * 10000)) / 100.0;

			if (pw != null) pw.println(prefixStr + displayStr);
			return f1;
		}

		public double getF1() {
			return display(false, null);
		}

		/**
		 * Returns F1
		 * 
		 * @param verbose
		 * @return
		 */
		public double display(boolean verbose) {
			return display(verbose, new PrintWriter(System.out, true));
		}

		/**
		 * Returns F1
		 * 
		 * @param verbose
		 * @param pw
		 * @return
		 */
		public double display(boolean verbose, PrintWriter pw) {
			return displayPRF(str + " [Average] ", correctEvents, guessedEvents, goldEvents, exact, total, pw);
		}
	}

	public static class LabeledConstituentEval<L> extends AbstractEval<L, LabeledConstituent<L>>
	{

		Set<L> labelsToIgnore;

		Set<L> punctuationTags;

		static <L> Tree<L> stripLeaves(Tree<L> tree) {
			if (tree.isLeaf()) return null;
			if (tree.isPreTerminal()) return new Tree<L>(tree.getLabel());
			List<Tree<L>> children = new ArrayList<Tree<L>>();
			for (Tree<L> child : tree.getChildren()) {
				children.add(stripLeaves(child));
			}
			return new Tree<L>(tree.getLabel(), children);
		}

		public Set<LabeledConstituent<L>> makeObjects(Tree<L> tree) {
			Tree<L> noLeafTree = stripLeaves(tree);
			Set<LabeledConstituent<L>> set = new HashSet<LabeledConstituent<L>>();
			addConstituents(noLeafTree, set, 0);
			return set;
		}

		private int addConstituents(Tree<L> tree, Set<LabeledConstituent<L>> set, int start) {
			if (tree.isLeaf()) {
				if (punctuationTags.contains(tree.getLabel()))
					return 0;
				else
					return 1;
			}
			int end = start;
			for (Tree<L> child : tree.getChildren()) {
				int childSpan = addConstituents(child, set, end);
				end += childSpan;
			}
			L label = tree.getLabel();
			if (!labelsToIgnore.contains(label)) {
				set.add(new LabeledConstituent<L>(label, start, end));
			}
			return end - start;
		}

		public LabeledConstituentEval(Set<L> labelsToIgnore, Set<L> punctuationTags) {
			this.labelsToIgnore = labelsToIgnore;
			this.punctuationTags = punctuationTags;
		}

	}

	public static void main(String[] args) throws Throwable {
		Tree<String> goldTree = (new Trees.PennTreeReader(new StringReader("(ROOT (S (NP (DT the) (NN can)) (VP (VBD fell))))"))).next();
		Tree<String> guessedTree = (new Trees.PennTreeReader(new StringReader("(ROOT (S (NP (DT the)) (VP (MB can) (VP (VBD fell)))))"))).next();
		LabeledConstituentEval<String> eval = new LabeledConstituentEval<String>(Collections.singleton("ROOT"), new HashSet<String>());
		eval.evaluate(guessedTree, goldTree);
		eval.display(true);
		System.out.println("XXX " + eval.evaluateF1(guessedTree, goldTree));
	}
}
