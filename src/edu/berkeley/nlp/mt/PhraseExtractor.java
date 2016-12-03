package edu.berkeley.nlp.mt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.assignments.align.AlignmentTester.IntPhrasePair;
import edu.berkeley.nlp.mt.Alignment.Link;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Pair;
import edu.berkeley.nlp.util.StrUtils;
import edu.berkeley.nlp.util.StringIndexer;

public class PhraseExtractor
{

	private int[][] getPruningMask(Alignment al, int srcLength, int trgLength) {

		boolean[][] alignmentsScratch = new boolean[srcLength][trgLength];
		CollectionUtils.fill(alignmentsScratch, false);
		for (Pair<Integer, Integer> link : al.getSureAlignments()) {
			alignmentsScratch[link.getSecond()][link.getFirst()] = true;
		}

		int[][] pruningMaskScratch = new int[srcLength + 1][trgLength + 1];
		CollectionUtils.fill(pruningMaskScratch, 0);

		for (int i = 1; i <= trgLength; ++i) {
			for (int j = 1; j <= srcLength; ++j) {
				if (i == 0 && j == 0) continue;
				final int subSpan1 = j == 0 ? 0 : pruningMaskScratch[j - 1][i];
				final int subSpan2 = i == 0 ? 0 : pruningMaskScratch[j][i - 1];
				final int subSpan3 = pruningMaskScratch[j - 1][i - 1];
				//				final AlignableObject tree = en(i);
				//				int treeOrderI = treeToIntMap.get(((NodeAlignableObject) tree).getNode());

				pruningMaskScratch[j][i] = subSpan1 + subSpan2 - subSpan3 + (alignmentsScratch[j - 1][i - 1] ? 1 : 0);
			}
		}
		return pruningMaskScratch;
	}

	private static String[] toStringArray(int[] english, StringIndexer lexIndexer) {
		String[] strings = new String[english.length];
		int k = 0;
		for (int i : english) {
			strings[k++] = lexIndexer == null ? ("" + i) : lexIndexer.get(i).toString();
		}
		return strings;
	}

	private boolean pruned(int[][] pruningMask, int srcStart, int srcEnd, int trgStart, int trgEnd, int numWords, int J) {

		int north = pruningMask[srcStart][trgEnd] - pruningMask[srcStart][trgStart];
		if (north >= 1) return true;
		int west = pruningMask[srcEnd][trgStart] - pruningMask[srcStart][trgStart];
		if (west >= 1) return true;
		int center = pruningMask[srcEnd][trgEnd] - pruningMask[srcStart][trgStart] - north - west;
		assert center >= 0;
		if (center < 1) return true;// check if there's actually an alignment

		int northEastCorner = pruningMask[srcStart][numWords] - pruningMask[srcStart][trgEnd];
		int east = pruningMask[srcEnd][numWords] - pruningMask[srcEnd][trgEnd] - northEastCorner;
		if (east >= 1) return true;
		int southWestCorner = pruningMask[J][trgStart] - pruningMask[srcEnd][trgStart];
		int south = pruningMask[J][trgEnd] - pruningMask[srcEnd][trgEnd] - southWestCorner;
		if (south >= 1) return true;

		assert north >= 0;
		assert south >= 0;
		assert east >= 0;
		assert west >= 0;
		if (north + south + west + east >= 1) return true;

		return false;
	}

	public List<IntPhrasePair> extract(Alignment al, SentencePair sentencePair, int maxPhraseSize, StringIndexer eWordIndexer, StringIndexer fWordIndexer,
		int maxNumUnaligned) {

		List<String> englishSentence = sentencePair.getEnglishWords();
		List<IntPhrasePair> ret = new ArrayList<IntPhrasePair>();
		List<List<Link>> enLinks = new ArrayList<List<Link>>();
		for (int i = 0; i < englishSentence.size(); ++i) {
			enLinks.add(al.getAlignmentsToEnglish(i));
		}
		int x = 0;
		final List<String> foreignSentence = sentencePair.getFrenchWords();
		final int foreignLength = foreignSentence.size();
		final int enLength = englishSentence.size();
		int[][] mask = getPruningMask(al, foreignLength, enLength);
		for (int enStart = 0; enStart < enLength; ++enStart) {
			for (int enEnd = enStart + 1; enEnd <= Math.min(enLength, enStart + maxPhraseSize); ++enEnd) {
				for (int frStart = 0; frStart < foreignLength; ++frStart) {
					for (int frEnd = frStart + 1; frEnd <= Math.min(foreignLength, frStart + maxPhraseSize); ++frEnd) {
						if (pruned(mask, frStart, frEnd, enStart, enEnd, enLength, foreignLength)) {
							continue;
						}
						String[] englishString = englishSentence.subList(enStart, enEnd).toArray(new String[enEnd - enStart]);
						int[] english = toArray(englishString, eWordIndexer);
						int[] foreign = toArray(foreignSentence.subList(frStart, frEnd).toArray(new String[frEnd - frStart]), fWordIndexer);
						int[] enStarts = new int[english.length];
						int[] enEnds = new int[english.length];
						boolean[] englishIsAligned = new boolean[english.length];
						Arrays.fill(englishIsAligned, false);
						boolean[] foreignIsAligned = new boolean[foreign.length];
						Arrays.fill(foreignIsAligned, false);
						for (int i = 0; i < english.length; ++i) {
							int min = Integer.MAX_VALUE;
							int max = Integer.MIN_VALUE;
							int en = enStart + i;

							for (Link link : al.getAlignmentsToEnglish(en)) {
								englishIsAligned[i] = true;
								foreignIsAligned[link.fr - frStart] = true;
								min = Math.min(min, link.fr);
								max = Math.max(max, link.fr);
							}
							if (min == Integer.MAX_VALUE && max == Integer.MIN_VALUE) {
								min = max = frStart;
							} else {
								max++;
							}
							enStarts[i] = min - frStart;
							enEnds[i] = max - frStart;
						}
						int numEnUnalignedLeft = 0;
						for (int i = 0; !englishIsAligned[i]; ++i) {
							numEnUnalignedLeft++;
						}
						int numEnUnalignedRight = 0;
						for (int i = englishIsAligned.length - 1; !englishIsAligned[i]; --i) {
							numEnUnalignedRight++;
						}
						int numFrUnalignedLeft = 0;
						for (int i = 0; !foreignIsAligned[i]; ++i) {
							numFrUnalignedLeft++;
						}
						int numFrUnalignedRight = 0;
						for (int i = foreignIsAligned.length - 1; !foreignIsAligned[i]; --i) {
							numFrUnalignedRight++;
						}
						int numUnaligned = numEnUnalignedLeft + numEnUnalignedRight + numFrUnalignedLeft + numFrUnalignedRight;
						if (numUnaligned > maxNumUnaligned) continue;
						x++;
						IntPhrasePair pair = new IntPhrasePair(foreign, english);
						ret.add(pair);
					}
				}
			}
		}
		return ret;

	}

	private static int[] toArray(final String[] ngram, StringIndexer eWordIndexer) {
		int[] ngramArray = new int[ngram.length];
		for (int w = 0; w < ngramArray.length; ++w) {
			ngramArray[w] = eWordIndexer.addAndGetIndex(ngram[w]);
		}
		return ngramArray;
	}
}