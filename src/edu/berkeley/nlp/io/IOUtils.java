package edu.berkeley.nlp.io;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;


/**
 * Utilities for getting files recursively, with a filter.
 * 
 * @author Dan Klein
 */
public class IOUtils
{
	public static List<File> getFilesUnder(String path, FileFilter fileFilter) {
		File root = new File(path);
		List<File> files = new ArrayList<File>();
		addFilesUnder(root, files, fileFilter);
		return files;
	}

	private static void addFilesUnder(File root, List<File> files, FileFilter fileFilter) {
		if (!fileFilter.accept(root)) return;
		if (root.isFile()) {
			files.add(root);
			return;
		}
		if (root.isDirectory()) {
			File[] children = root.listFiles();
			for (int i = 0; i < children.length; i++) {
				File child = children[i];
				addFilesUnder(child, files, fileFilter);
			}
		}
	}

	public static Iterator<String> lineIterator(String path) throws IOException {
		final BufferedReader reader = IOUtils.openIn(path);
		return lineIterator(reader);
	}

	/**
	 * @param reader
	 * @return
	 */
	public static Iterator<String> lineIterator(final BufferedReader reader) {
		return new Iterator<String>()
		{

			private String line;

			public boolean hasNext() {
				// TODO Auto-generated method stub
				try {
					return nextLine();
				} catch (Exception e) {
					e.printStackTrace();
				}
				return false;
			}

			private boolean nextLine() throws IOException {
				if (line != null) { return true; }
				line = reader.readLine();
				return line != null;
			}

			public String next() {
				// TODO Auto-generated method stub
				try {
					nextLine();
					String retLine = line;
					line = null;
					return retLine;
				} catch (IOException e) {
					throw new RuntimeException();
				}
			}

			public void remove() {
				// TODO Auto-generated method stub
				throw new RuntimeException("remove() not supported");
			}

		};
	}

	public static BufferedReader openIn(String path) throws IOException {
		return openIn(new File(path));
	}

	public static BufferedReader openIn(File path) throws IOException {
		InputStream is = new FileInputStream(path);
		if (path.getName().endsWith(".gz")) is = new GZIPInputStream(is);
		return new BufferedReader(getReader(is));
	}

	public static BufferedReader getReader(InputStream in) throws IOException {
		return new BufferedReader(new InputStreamReader(in, "UTF-8"));
	}

	public static BufferedReader openInHard(String path) {
		return openInHard(new File(path));
	}

	public static BufferedReader openInHard(File path) {
		try {
			return openIn(path);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	// openOut
	public static PrintWriter openOut(String path) throws IOException {
		return openOut(new File(path));
	}

	public static PrintWriter openOut(File path) throws IOException {
		OutputStream os = new FileOutputStream(path);
		if (path.getName().endsWith(".gz")) os = new GZIPOutputStream(os);
		return new PrintWriter(new OutputStreamWriter(os, "UTF-8"), true);
	}

	public static PrintWriter openOutEasy(String path) {

		return openOutEasy(new File(path));
	}

	public static PrintWriter openOutEasy(File path) {
		if (path == null) return null;
		try {
			return openOut(path);
		} catch (Exception e) {
			return null;
		}
	}

	public static PrintWriter openOutHard(String path) {
		return openOutHard(new File(path));
	}

	public static PrintWriter openOutHard(File path) {
		try {
			return openOut(path);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	// READING AND WRITING SERIALIZED OBJECTS

  public static void save(Serializable  obj, String path) {
    if (!new File(path).getParentFile().canWrite()) {
      throw new RuntimeException("Can't write to " + path); 
    }
    if (path.endsWith(".gz")) {
      saveGz(obj, path);
    } else {
      saveNonGz(obj, path);
    }
  }
  
  public static void saveNonGz(Serializable obj, String path) {
    try {
      FileOutputStream fileOut = new FileOutputStream(path);
      ObjectOutputStream out = new ObjectOutputStream(fileOut);
      out.writeObject(obj);
      System.out.println("Wrote to " + path);
      out.close();
      fileOut.close();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
  
  public static void saveGz(Serializable obj, String path) {
    try {
      BufferedOutputStream  fileOut = new BufferedOutputStream(new FileOutputStream(path));
      GZIPOutputStream gzOutputStream = new GZIPOutputStream(fileOut);
      ObjectOutputStream out = new ObjectOutputStream(gzOutputStream);
      out.writeObject(obj);
      System.out.println("Wrote to " + path);
      out.close();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
  
  public static Object load(String path) {
    if (!new File(path).canRead()) {
      throw new RuntimeException("Can't read from " + path); 
    }
    if (path.endsWith(".gz")) {
      return loadGz(path);
    } else {
      return loadNonGz(path);
    }
  }
  
  public static Object loadNonGz(String path) {
    Object obj = null;
    try {
      FileInputStream fileIn = new FileInputStream(new File(path));
      ObjectInputStream in = new ObjectInputStream(fileIn);
      obj = in.readObject();
      System.out.println("Object read from " + path);
      in.close();
      fileIn.close();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return obj;
  }
  
  public static Object loadGz(String path) {
    Object obj = null;
    try {
      BufferedInputStream fileIn = new BufferedInputStream(new FileInputStream(path));
      GZIPInputStream gzipInputStream = new GZIPInputStream(fileIn);
      ObjectInputStream in = new ObjectInputStream(gzipInputStream);
      obj = in.readObject();
      System.out.println("Object read from " + path);
      in.close();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return obj;
  }
}
