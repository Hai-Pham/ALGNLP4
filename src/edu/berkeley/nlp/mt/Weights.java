package edu.berkeley.nlp.mt;

import java.io.File;
import java.io.IOException;

import edu.berkeley.nlp.io.IOUtils;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Counter;

public class Weights
{

	public static Counter<String> readWeightsFile(File weightsFile) {
		Counter<String> ret = new Counter<String>();
		try {
			for (String line : CollectionUtils.iterable(IOUtils.lineIterator(weightsFile.getPath()))) {
				if (line.trim().length() == 0) continue;
				String[] parts = line.trim().split("\t");
				ret.setCount(parts[0].intern(), Double.parseDouble(parts[1]));

			}
		} catch (NumberFormatException e) {
			System.err.println("Error reading weights file " + weightsFile);
			throw new RuntimeException(e);

		} catch (IOException e) {
			System.err.println("Error reading weights file " + weightsFile);
			throw new RuntimeException(e);

		}
		return ret;
	}

}
