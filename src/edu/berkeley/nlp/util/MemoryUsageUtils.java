package edu.berkeley.nlp.util;

public class MemoryUsageUtils
{

	/**
	 * 
	 */
	public static void printMemoryUsage() {
		System.gc();
		System.gc();
		System.gc();
		System.out.println("Memory usage is " + getUsedMemoryStr());
	}

	public static String bytesToString(long b) {
		double gb = (double) b / (1024 * 1024 * 1024);
		if (gb >= 1) return gb >= 10 ? (int) gb + "G" : round(gb, 1) + "G";
		double mb = (double) b / (1024 * 1024);
		if (mb >= 1) return mb >= 10 ? (int) mb + "M" : round(mb, 1) + "M";
		double kb = (double) b / (1024);
		if (kb >= 1) return kb >= 10 ? (int) kb + "K" : round(kb, 1) + "K";
		return b + "";
	}

	public static double round(double x, int numPlaces) {
		double scale = Math.pow(10, numPlaces);
		return Math.round(x * scale) / scale;
	}

	public static String getUsedMemoryStr() {
		long totalMem = Runtime.getRuntime().totalMemory();
		long freeMem = Runtime.getRuntime().freeMemory();
		return bytesToString(totalMem - freeMem);
	}

}
