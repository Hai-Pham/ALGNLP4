package edu.berkeley.nlp.util;

public interface IterableWithSize<T> extends Iterable<T>
{
	public int size();
}