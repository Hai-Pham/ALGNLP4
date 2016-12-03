package edu.berkeley.nlp.io;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class SentenceCollection implements Iterable<List<String>>
{
	public static class SentenceIterator implements Iterator<List<String>>
	{

	  int maxSentences;
	  int sentencesReturned;
		Iterator<String> reader;

		public boolean hasNext() {
			return reader.hasNext() && (maxSentences < 0 || sentencesReturned < maxSentences);
		}

		public List<String> next() {
			String line = reader.next();
			String[] words = line.split("\\s+");
			List<String> sentence = new ArrayList<String>();
			for (int i = 0; i < words.length; i++) {
				String word = words[i];
				sentence.add(word.toLowerCase());
			}
			sentencesReturned += 1;
			return sentence;
		}

		public void remove() {
			throw new UnsupportedOperationException();
		}

		public SentenceIterator(Iterator<String> reader, int maxSentences) {
			this.reader = reader;
			this.maxSentences = maxSentences;
			this.sentencesReturned = 0;
		}
	}

	String fileName;
	int maxSentences;

	public Iterator<List<String>> iterator() {
		try {

			return new SentenceIterator(IOUtils.lineIterator(fileName), maxSentences);
		} catch (FileNotFoundException e) {
			System.err.println("Problem with SentenceIterator for " + fileName);
			throw new RuntimeException(e);
		} catch (IOException e) {
			System.err.println("Problem with SentenceIterator for " + fileName);
			throw new RuntimeException(e);

		}
	}

	public SentenceCollection(String fileName, int maxSentences) {
		this.fileName = fileName;
		this.maxSentences = maxSentences;
	}

	public static class Reader
	{
		public static Iterable<List<String>> readSentenceCollection(String fileName) {
			return new SentenceCollection(fileName, -1);
		}

    public static Iterable<List<String>> readSentenceCollection(String fileName, int maxSentences) {
      return new SentenceCollection(fileName, maxSentences);
    }
	}

}